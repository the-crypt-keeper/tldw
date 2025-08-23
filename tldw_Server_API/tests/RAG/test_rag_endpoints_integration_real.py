"""
Real integration tests for RAG endpoints without mocking.

These tests interact with the actual RAG service and databases
to verify end-to-end functionality.
"""

import json
import pytest
import asyncio
import tempfile
import shutil
import os
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from unittest import mock

from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.api.v1.endpoints.rag_api import router as rag_router


@pytest.fixture(scope="module", autouse=True)
def disable_csrf():
    """Disable CSRF for testing."""
    original_csrf = settings.get("CSRF_ENABLED", None)
    settings["CSRF_ENABLED"] = False
    yield
    if original_csrf is not None:
        settings["CSRF_ENABLED"] = original_csrf
    else:
        settings.pop("CSRF_ENABLED", None)


@pytest.fixture(scope="function")
async def setup_test_data():
    """Set up test data in the actual database locations the service expects."""
    # Get the user database directory (user 1 for single-user mode, from SINGLE_USER_FIXED_ID)
    user_db_dir = Path(settings.get("USER_DB_BASE_DIR")) / "1"
    user_db_dir.mkdir(parents=True, exist_ok=True)
    
    # Create media database with test data
    media_db_path = user_db_dir / "user_media_library.sqlite"
    media_db = MediaDatabase(str(media_db_path), client_id="1")
    
    # Add sample media items
    sample_media = [
        {
            "title": "Introduction to RAG Systems",
            "content": "Retrieval-Augmented Generation (RAG) combines the power of retrieval systems with generative models. This allows AI systems to access external knowledge bases to provide more accurate and up-to-date information. RAG is essential for building intelligent systems.",
            "url": "https://example.com/rag-intro",
            "media_type": "article",
            "author": "AI Expert",
            "ingestion_date": "2024-01-15"
        },
        {
            "title": "Machine Learning Fundamentals",
            "content": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It includes supervised learning, unsupervised learning, and reinforcement learning approaches.",
            "url": "https://example.com/ml-basics",
            "media_type": "article",
            "author": "ML Teacher",
            "ingestion_date": "2024-02-01"
        },
        {
            "title": "Vector Databases Tutorial",
            "content": "Vector databases are specialized databases designed to store and query high-dimensional vector embeddings. They are essential for semantic search and RAG applications, enabling efficient similarity search at scale.",
            "url": "https://example.com/vector-db",
            "media_type": "video",
            "author": "Database Expert",
            "ingestion_date": "2024-03-10"
        }
    ]
    
    media_ids = []
    for item in sample_media:
        media_id = media_db.add_media_with_keywords(**item)
        media_ids.append(media_id)
    
    # Create ChaChaNotes database with test data
    chacha_db_dir = user_db_dir / "chachanotes_user_dbs"
    chacha_db_dir.mkdir(parents=True, exist_ok=True)
    chacha_db_path = chacha_db_dir / "user_chacha_notes_rag.sqlite"
    chacha_db = CharactersRAGDB(str(chacha_db_path), client_id="1")
    
    # Add sample notes
    note1_id = chacha_db.add_note(
        title="RAG Implementation Best Practices",
        content="Key points for RAG implementation: 1) Choose appropriate embedding model, 2) Optimize chunk size for context, 3) Implement proper re-ranking for relevance"
    )
    
    note2_id = chacha_db.add_note(
        title="Meeting Notes - AI Strategy",
        content="Discussed moving to RAG-based system for better accuracy. Need to evaluate different vector databases and embedding models for production deployment."
    )
    
    # Clear any cached services to ensure fresh start
    await rag_service_manager.cleanup_expired()
    
    yield {
        "media_db": media_db,
        "chacha_db": chacha_db,
        "media_ids": media_ids,
        "note_ids": [note1_id, note2_id]
    }
    
    # Cleanup after test
    media_db.close_connection()
    chacha_db.close_connection()
    
    # Note: In a real test environment, you might want to backup and restore
    # the original databases instead of modifying them directly


@pytest.fixture
def auth_headers():
    """Create authentication headers for the test."""
    return {"X-API-KEY": settings.get("API_KEY", "default-secret-key-for-single-user")}


@pytest.fixture
async def async_client():
    """Create an async test client."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client


class TestRAGSearchRealIntegration:
    """Real integration tests for the search endpoint without mocking."""
    
    @pytest.mark.asyncio
    async def test_search_finds_real_data(self, async_client, auth_headers, setup_test_data):
        """Test that search actually finds the data we inserted."""
        response = await async_client.post(
            "/api/v1/rag/search",
            json={
                "query": "RAG implementation",
                "search_type": "fulltext",  # Use fulltext for predictable results
                "databases": ["media_db", "notes"],
                "limit": 10
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "results" in data
        assert "total_results" in data
        assert "query_id" in data
        assert "search_type_used" in data
        
        # Check that we found some results
        results = data["results"]
        assert isinstance(results, list)
        assert data["total_results"] > 0, "Should find at least one result for 'RAG implementation'"
        
        # Verify result structure
        if len(results) > 0:
            first_result = results[0]
            assert "id" in first_result
            assert "title" in first_result
            assert "content" in first_result
            assert "score" in first_result
            assert "source" in first_result
    
    @pytest.mark.asyncio
    async def test_search_with_keywords(self, async_client, auth_headers, setup_test_data):
        """Test search with keyword filtering."""
        response = await async_client.post(
            "/api/v1/rag/search",
            json={
                "query": "machine learning",
                "search_type": "fulltext",
                "databases": ["media_db"],
                "keywords": ["learning", "AI"],
                "limit": 5
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should find the ML fundamentals article
        assert data["total_results"] > 0
        results = data["results"]
        
        # Check if results contain our keywords
        for result in results:
            content_lower = result["content"].lower()
            # At least one keyword should be present
            assert any(kw.lower() in content_lower for kw in ["learning", "ai", "machine"])
    
    @pytest.mark.asyncio
    async def test_search_notes_database(self, async_client, auth_headers, setup_test_data):
        """Test searching specifically in notes database."""
        response = await async_client.post(
            "/api/v1/rag/search",
            json={
                "query": "best practices",
                "search_type": "fulltext",
                "databases": ["notes"],
                "limit": 10
            },
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        
        # Should find the note about RAG best practices
        assert data["total_results"] > 0
        
        # Verify results are from notes
        for result in data["results"]:
            assert result["source"] in ["notes", "NOTES", "DataSource.NOTES"]


class TestRAGAgentRealIntegration:
    """Real integration tests for the agent endpoint without mocking."""
    
    @pytest.mark.asyncio
    async def test_agent_basic_question(self, async_client, auth_headers, setup_test_data):
        """Test basic Q&A with the agent using real data."""
        response = await async_client.post(
            "/api/v1/rag/agent",
            json={
                "message": "What is RAG and how does it work?",
                "search_databases": ["media_db", "notes"]
            },
            headers=auth_headers
        )
        
        if response.status_code != 200:
            print(f"Response error: {response.text}")
        assert response.status_code == 200
        data = response.json()
        
        # Verify response structure
        assert "response" in data
        assert "conversation_id" in data
        assert "sources" in data
        
        # Response should be present (even if fallback generator is used)
        assert len(data["response"]) > 0
        
        # Sources might be empty if no relevant content found or using FallbackGenerator
        # Just verify the sources field exists and is a list
        assert isinstance(data["sources"], list)
        
        # If we have sources, check their structure
        if len(data["sources"]) > 0:
            source = data["sources"][0]
            assert "title" in source
            assert "content" in source
            assert "database" in source
            assert "relevance_score" in source
    
    @pytest.mark.asyncio
    async def test_agent_with_conversation_context(self, async_client, auth_headers, setup_test_data):
        """Test agent maintains conversation context."""
        # First message
        response1 = await async_client.post(
            "/api/v1/rag/agent",
            json={
                "message": "What are vector databases?",
                "search_databases": ["media_db"]
            },
            headers=auth_headers
        )
        
        assert response1.status_code == 200
        data1 = response1.json()
        conversation_id = data1["conversation_id"]
        
        # Follow-up message using same conversation
        response2 = await async_client.post(
            "/api/v1/rag/agent",
            json={
                "message": "How are they used in RAG systems?",
                "conversation_id": conversation_id,
                "search_databases": ["media_db", "notes"]
            },
            headers=auth_headers
        )
        
        assert response2.status_code == 200
        data2 = response2.json()
        
        # Should maintain same conversation
        assert data2["conversation_id"] == conversation_id
        
        # Response should exist (contextual response requires actual LLM)
        assert len(data2["response"]) > 0
        # With FallbackGenerator, we can't test contextual understanding
        # but we can verify the conversation flow works


class TestRAGServiceCachingReal:
    """Test caching behavior with real services."""
    
    @pytest.mark.asyncio
    async def test_service_reuse_across_requests(self, async_client, auth_headers, setup_test_data):
        """Test that RAG services are cached and reused."""
        # Clear cache first
        await rag_service_manager.cleanup_expired()
        
        # First request - creates new service
        response1 = await async_client.post(
            "/api/v1/rag/search",
            json={"query": "test", "limit": 1},
            headers=auth_headers
        )
        assert response1.status_code == 200
        
        # Check service was cached (user 1 for single-user mode, from SINGLE_USER_FIXED_ID)
        assert 1 in rag_service_manager._cache
        service1 = rag_service_manager._cache[1]['service']
        
        # Second request - should reuse service
        response2 = await async_client.post(
            "/api/v1/rag/search",
            json={"query": "another test", "limit": 1},
            headers=auth_headers
        )
        assert response2.status_code == 200
        
        # Verify same service instance was used
        assert 1 in rag_service_manager._cache
        service2 = rag_service_manager._cache[1]['service']
        assert service2 is service1


class TestErrorScenariosReal:
    """Test real error scenarios without mocking."""
    
    @pytest.mark.asyncio
    async def test_invalid_search_parameters(self, async_client, auth_headers):
        """Test validation of search parameters."""
        response = await async_client.post(
            "/api/v1/rag/search",
            json={
                "query": "",  # Empty query should fail
                "limit": -1,  # Invalid limit
            },
            headers=auth_headers
        )
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_invalid_database_name(self, async_client, auth_headers):
        """Test error when specifying invalid database."""
        response = await async_client.post(
            "/api/v1/rag/search",
            json={
                "query": "test",
                "databases": ["invalid_db"]  # This should fail validation
            },
            headers=auth_headers
        )
        
        assert response.status_code == 422
        error_detail = response.json()
        assert "Invalid database" in str(error_detail)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])