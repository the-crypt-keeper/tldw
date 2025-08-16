"""
Fixed integration tests for RAG v2 endpoints.

These tests properly test the endpoints without excessive mocking,
following the pattern from test_rag_endpoints_integration.py
"""

import json
import pytest
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from uuid import uuid4
from unittest.mock import Mock, AsyncMock

from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.api.v1.endpoints.rag_v2 import rag_service_manager
from tldw_Server_API.app.core.config import settings


class TestRAGV2Endpoints:
    """Fixed integration tests for RAG v2 endpoints."""
    
    @classmethod
    def setup_class(cls):
        """Set up test fixtures."""
        # Disable CSRF for testing
        cls.original_csrf = settings.get("CSRF_ENABLED", None)
        settings["CSRF_ENABLED"] = False
        
        cls.client = TestClient(app)
        cls.test_user = User(
            id=0,  # Using 0 for single-user mode
            username="test_user",
            email="test@example.com",
            is_active=True
        )
        cls.auth_headers = {"X-API-KEY": "default-secret-key-for-single-user"}
        
        # Create temp directory for test databases
        cls.test_dir = tempfile.mkdtemp(prefix="rag_v2_test_")
        cls.user_base_dir = Path(cls.test_dir) / "user_databases"
        cls.user_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup test databases in the expected location
        cls.setup_test_databases()
    
    @classmethod
    def teardown_class(cls):
        """Clean up test fixtures."""
        # Restore CSRF setting
        if cls.original_csrf is not None:
            settings["CSRF_ENABLED"] = cls.original_csrf
        else:
            settings.pop("CSRF_ENABLED", None)
        
        # Clean up test directory
        shutil.rmtree(cls.test_dir, ignore_errors=True)
        
        # Clear any cached services
        import asyncio
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(rag_service_manager.cleanup_expired())
        loop.close()
    
    @classmethod
    def setup_test_databases(cls):
        """Set up test databases with sample data."""
        # Create user directory structure
        user_dir = cls.user_base_dir / "0"
        user_dir.mkdir(parents=True, exist_ok=True)
        
        # Create media database
        media_db_path = user_dir / "user_media_library.sqlite"
        media_db = MediaDatabase(str(media_db_path), client_id="0")
        
        # Add sample media
        sample_media = [
            {
                "title": "RAG Overview",
                "content": "Retrieval-Augmented Generation (RAG) combines retrieval and generation.",
                "url": "https://example.com/rag",
                "media_type": "article",
                "author": "AI Expert",
                "ingestion_date": datetime.now().isoformat()
            },
            {
                "title": "Machine Learning Concepts",
                "content": "Machine learning enables systems to learn from data.",
                "url": "https://example.com/ml",
                "media_type": "article",
                "author": "ML Expert",
                "ingestion_date": datetime.now().isoformat()
            }
        ]
        
        for item in sample_media:
            media_db.add_media_with_keywords(**item)
        
        media_db.close_connection()
        
        # Create ChaChaNotes database
        chacha_dir = user_dir / "chachanotes_user_dbs"
        chacha_dir.mkdir(parents=True, exist_ok=True)
        chacha_db_path = chacha_dir / "user_chacha_notes_rag.sqlite"
        chacha_db = CharactersRAGDB(str(chacha_db_path), client_id="0")
        
        # Add sample character and conversation
        char_data = {
            'name': 'Test Assistant',
            'description': 'A helpful AI assistant',
            'personality': 'Professional',
            'tags': json.dumps(['test']),
            'creator': 'test',
            'client_id': '0'
        }
        char_id = chacha_db.add_character_card(char_data)
        
        # Add conversation
        conv_id = str(uuid4())
        conv_data = {
            'id': conv_id,
            'character_id': char_id,
            'title': 'Test Conversation',
            'client_id': '0'
        }
        chacha_db.add_conversation(conv_data)
        
        # Add messages
        messages = [
            {"conversation_id": conv_id, "sender": "user", "content": "Previous question"},
            {"conversation_id": conv_id, "sender": "assistant", "content": "Previous answer"}
        ]
        for msg in messages:
            chacha_db.add_message(msg)
        
        chacha_db.close_connection()
        
        # Store the conversation ID for later use
        cls.test_conversation_id = conv_id
    
    # ============= Search Tests =============
    
    def test_simple_search_semantic(self):
        """Test simple semantic search without mocking."""
        response = self.client.post(
            "/api/v1/rag/search",
            headers=self.auth_headers,
            json={
                "query": "machine learning concepts",
                "search_type": "semantic",
                "databases": ["media_db"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
        assert "total_results" in data
        assert data["total_results"] >= 0
    
    def test_advanced_search_with_filters(self):
        """Test advanced search without mocking."""
        response = self.client.post(
            "/api/v1/rag/search/advanced",
            headers=self.auth_headers,
            json={
                "query": "advanced query",
                "search_config": {
                    "search_type": "hybrid",
                    "limit": 20,
                    "databases": ["media_db"]
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert isinstance(data["results"], list)
    
    # ============= Agent Tests =============
    
    def test_simple_agent_qa(self):
        """Test simple agent Q&A without mocking."""
        # Mock only the LLM call since we don't want to make real API calls
        with MockLLMResponse("RAG is a technique that combines information retrieval with text generation."):
            response = self.client.post(
                "/api/v1/rag/agent",
                headers=self.auth_headers,
                json={
                    "message": "What is RAG?",
                    "search_databases": ["media_db"],
                    "model": "gpt-4"
                }
            )
        
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "conversation_id" in data
        assert "sources" in data
    
    def test_simple_agent_with_conversation(self):
        """Test simple agent with conversation context."""
        # Use the conversation ID from setup
        with MockLLMResponse("Follow-up answer based on context."):
            response = self.client.post(
                "/api/v1/rag/agent",
                headers=self.auth_headers,
                json={
                    "message": "Tell me more",
                    "conversation_id": self.test_conversation_id,
                    "search_databases": ["media_db"]
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert data["conversation_id"] == self.test_conversation_id
    
    def test_advanced_agent_research_mode(self):
        """Test advanced agent in research mode."""
        with MockLLMResponse("Comprehensive research answer."):
            response = self.client.post(
                "/api/v1/rag/agent/advanced",
                headers=self.auth_headers,
                json={
                    "message": "Research RAG techniques",
                    "mode": "research",
                    "search_config": {
                        "databases": ["media_db"],
                        "search_type": "hybrid",
                        "limit": 10
                    },
                    "generation_config": {
                        "model": "gpt-4",
                        "temperature": 0.7
                    }
                }
            )
        
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "sources" in data
        assert "mode_used" in data
        assert data["mode_used"] == "research"
    
    def test_advanced_agent_streaming(self):
        """Test advanced agent with streaming (non-streaming fallback)."""
        # Streaming is complex to test, so we test the non-streaming fallback
        with MockLLMResponse("Streaming response content."):
            response = self.client.post(
                "/api/v1/rag/agent/advanced",
                headers=self.auth_headers,
                json={
                    "message": "Stream this response",
                    "generation_config": {
                        "stream": False  # Use non-streaming for simplicity
                    }
                }
            )
        
        if response.status_code != 200:
            print(f"Response status: {response.status_code}")
            print(f"Response body: {response.text}")
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
    
    def test_search_error_handling(self):
        """Test error handling with invalid request."""
        response = self.client.post(
            "/api/v1/rag/search",
            headers=self.auth_headers,
            json={
                # Invalid: missing required 'query' field
                "search_type": "semantic"
            }
        )
        
        # Should return validation error
        assert response.status_code == 422
        data = response.json()
        assert "detail" in data


class MockLLMResponse:
    """Context manager to mock LLM responses."""
    
    def __init__(self, response_text):
        self.response_text = response_text
        self.original_func = None
    
    def __enter__(self):
        """Mock the LLM call."""
        from tldw_Server_API.app.core.RAG.rag_service.integration import RAGService
        import unittest.mock as mock
        
        # Create a mock for generate_answer method that returns proper format
        async def mock_generate(*args, **kwargs):
            # Check if called with 'messages' parameter (advanced agent)
            if 'messages' in kwargs:
                # Advanced agent expects a string response
                return self.response_text
            else:
                # Simple agent expects a dict with 'answer' key
                return {
                    "answer": self.response_text,
                    "sources": [],
                    "metadata": {}
                }
        
        # Patch the RAGService.generate_answer method
        self.patcher = mock.patch.object(RAGService, 'generate_answer', new=mock_generate)
        self.patcher.start()
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop the patch."""
        if hasattr(self, 'patcher'):
            self.patcher.stop()