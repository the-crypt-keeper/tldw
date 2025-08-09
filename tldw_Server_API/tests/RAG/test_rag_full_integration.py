# test_rag_full_integration.py
# Comprehensive integration tests for the RAG pipeline
import asyncio
import json
import os
import tempfile
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List
from unittest.mock import patch, Mock
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

from tldw_Server_API.app.main import app
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.config import settings
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB
from tldw_Server_API.app.core.RAG.rag_service.integration import RAGService
from tldw_Server_API.app.core.RAG.rag_service.config import RAGConfig
from tldw_Server_API.app.api.v1.endpoints.rag import _user_rag_services


class TestRAGFullIntegration:
    """Full integration tests for the RAG pipeline."""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup and cleanup for each test."""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.user_base_dir = Path(self.temp_dir) / "users"
        self.user_base_dir.mkdir(exist_ok=True)
        
        # Clear the RAG service cache
        _user_rag_services.clear()
        
        # Patch settings - settings is a dict-like object, so we need to mock differently
        self.original_user_db_dir = settings.get("USER_DB_BASE_DIR", "/tmp/users")
        settings["USER_DB_BASE_DIR"] = str(self.user_base_dir)
        
        yield
        
        # Cleanup
        settings["USER_DB_BASE_DIR"] = self.original_user_db_dir
        _user_rag_services.clear()
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @pytest.fixture
    def test_users(self):
        """Create test users."""
        return [
            User(id=1, username="user1", email="user1@test.com", is_active=True),
            User(id=2, username="user2", email="user2@test.com", is_active=True)
        ]
    
    @pytest.fixture
    def mock_auth(self, test_users):
        """Mock authentication to return test users."""
        async def get_user(user_id: int):
            return next((u for u in test_users if u.id == user_id), None)
        return get_user
    
    @pytest.fixture
    def auth_headers(self):
        """Get authentication headers for single-user mode."""
        return {"X-API-KEY": "default-secret-key-for-single-user"}
    
    def create_test_media_db(self, user_id: int) -> Path:
        """Create a test media database with sample data."""
        user_dir = self.user_base_dir / str(user_id)
        user_dir.mkdir(exist_ok=True)
        db_path = user_dir / "user_media_library.sqlite"
        
        # Create database using MediaDatabase to ensure proper schema
        media_db = MediaDatabase(str(db_path), client_id=str(user_id))
        
        # Add test media entries using the proper API
        test_data = [
            {
                "url": "https://example.com/rag-intro",
                "title": "Introduction to RAG",
                "content": "Retrieval-Augmented Generation (RAG) is a powerful technique that combines retrieval and generation.",
                "media_type": "article",
                "author": "AI Expert",
                "ingestion_date": datetime.now().isoformat(),
                "keywords": ["RAG", "retrieval", "generation", "AI"]
            },
            {
                "url": "https://example.com/ml-basics",
                "title": "Machine Learning Basics",
                "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "media_type": "video",
                "author": "ML Teacher",
                "ingestion_date": datetime.now().isoformat(),
                "keywords": ["machine learning", "AI", "data science"]
            },
            {
                "url": "https://example.com/python-guide",
                "title": "Python Programming Guide",
                "content": "Python is a versatile programming language used for web development, data science, and automation.",
                "media_type": "document",
                "author": "Code Master",
                "ingestion_date": datetime.now().isoformat(),
                "keywords": ["python", "programming", "automation"]
            },
            {
                "url": "https://example.com/dl-tutorial",
                "title": "Deep Learning Tutorial",
                "content": "Deep learning uses neural networks with multiple layers to progressively extract higher-level features.",
                "media_type": "video",
                "author": "DL Researcher",
                "ingestion_date": datetime.now().isoformat(),
                "keywords": ["deep learning", "neural networks", "AI"]
            },
            {
                "url": "https://example.com/nlp-intro",
                "title": "Natural Language Processing",
                "content": "NLP enables computers to understand, interpret, and generate human language in valuable ways.",
                "media_type": "article",
                "author": "NLP Expert",
                "ingestion_date": datetime.now().isoformat(),
                "keywords": ["NLP", "natural language", "AI"]
            }
        ]
        
        for entry in test_data:
            # Use the add_media_with_keywords method
            media_db.add_media_with_keywords(
                url=entry["url"],
                title=entry["title"],
                media_type=entry["media_type"],
                content=entry["content"],
                keywords=entry["keywords"],
                prompt="",
                transcription_model="test-model",
                author=entry["author"],
                ingestion_date=entry["ingestion_date"]
            )
        
        return db_path
    
    def create_test_chachanotes_db(self, user_id: int) -> Path:
        """Create a test ChaChaNotes database with sample data."""
        user_dir = self.user_base_dir / str(user_id)
        chacha_dir = user_dir / "chachanotes_user_dbs"
        chacha_dir.mkdir(exist_ok=True)
        db_path = chacha_dir / "user_chacha_notes_rag.sqlite"
        
        # Create database using CharactersRAGDB to ensure proper schema
        chacha_db = CharactersRAGDB(str(db_path), client_id=str(user_id))
        
        # Add test notes
        test_notes = [
            {
                "title": "RAG Implementation Notes",
                "content": "Important: RAG requires both a retriever and a generator component. The retriever finds relevant documents.",
                "keywords": "rag,implementation,retriever"
            },
            {
                "title": "Meeting Notes: AI Strategy",
                "content": "Discussed implementing RAG for our search system. Key benefits: better context, more accurate responses.",
                "keywords": "meeting,ai,strategy,rag"
            },
            {
                "title": "Research: Vector Databases",
                "content": "ChromaDB and Pinecone are popular choices for vector storage in RAG systems. ChromaDB is open source.",
                "keywords": "research,vector,database,chromadb"
            }
        ]
        
        for note in test_notes:
            # Add note (keywords are added to content for searchability)
            note_content = f"{note['content']}\n\nKeywords: {note['keywords']}"
            chacha_db.add_note(
                title=note["title"],
                content=note_content
            )
        
        return db_path
    
    def setup_user_environment(self, user_id: int):
        """Set up complete test environment for a user."""
        # Create databases
        media_db_path = self.create_test_media_db(user_id)
        chacha_db_path = self.create_test_chachanotes_db(user_id)
        
        # Create ChromaDB directory
        chroma_dir = self.user_base_dir / str(user_id) / "chroma"
        chroma_dir.mkdir(exist_ok=True)
        
        return {
            "media_db": media_db_path,
            "chacha_db": chacha_db_path,
            "chroma_dir": chroma_dir
        }
    
    @pytest.mark.asyncio
    async def test_search_endpoint_full_pipeline(self, test_users):
        """Test the search endpoint with real data."""
        # Setup environment for user1
        user1_env = self.setup_user_environment(test_users[0].id)
        
        # Mock authentication
        with patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user') as mock_auth:
            mock_auth.return_value = test_users[0]
            
            # Create actual database instances
            media_db = MediaDatabase(str(user1_env["media_db"]), client_id=str(test_users[0].id))
            chacha_db = CharactersRAGDB(str(user1_env["chacha_db"]), client_id=str(test_users[0].id))
            
            # Mock the RAG service to use our test databases
            from tldw_Server_API.app.api.v1.endpoints.rag import get_rag_service_for_user
            async def mock_get_rag_service():
                rag_service = RAGService(
                    config=RAGConfig(),
                    media_db_path=Path(user1_env["media_db"]),
                    chachanotes_db_path=Path(user1_env["chacha_db"]),
                    chroma_path=Path(user1_env["chroma_dir"]),
                    llm_handler=None
                )
                await rag_service.initialize()
                return rag_service
            
            # Mock database dependencies to return actual instances
            with patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user') as mock_media_db:
                with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user') as mock_chacha_db:
                    mock_media_db.return_value = media_db
                    mock_chacha_db.return_value = chacha_db
                    
                    # Override the RAG service creation
                    app.dependency_overrides[get_rag_service_for_user] = mock_get_rag_service
                    
                    try:
                        # Create test client with auth headers
                        headers = {"X-API-KEY": "default-secret-key-for-single-user"}
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            # Test 1: Basic search
                            response = await client.post(
                                "/api/v1/retrieval_agent/search",
                                json={
                                    "querystring": "RAG retrieval",
                                    "search_mode": "basic",
                                    "limit": 5
                                },
                                headers=headers
                            )
                            
                            assert response.status_code == 200
                            data = response.json()
                            assert "results" in data
                            assert len(data["results"]) > 0
                            assert any("RAG" in result["title"] or "RAG" in result["snippet"] for result in data["results"])
                            
                            # Test 2: Search with database selection
                            response = await client.post(
                                "/api/v1/retrieval_agent/search",
                                json={
                                    "querystring": "implementation",
                                    "search_databases": ["media_db", "notes"],
                                    "search_mode": "custom",
                                    "use_semantic_search": True
                                },
                                headers=headers
                            )
                            
                            assert response.status_code == 200
                            data = response.json()
                            assert len(data["results"]) > 0
                            
                            # Test 3: Search without date filters (ChromaDB issue with date filtering)
                            response = await client.post(
                                "/api/v1/retrieval_agent/search",
                                json={
                                    "querystring": "machine learning",
                                    "search_mode": "advanced"
                                },
                                headers=headers
                            )
                            
                            assert response.status_code == 200
                            data = response.json()
                            # Should find entries with machine learning
                            assert len(data["results"]) > 0
                    finally:
                        # Clean up dependency override
                        if get_rag_service_for_user in app.dependency_overrides:
                            del app.dependency_overrides[get_rag_service_for_user]
    
    @pytest.mark.asyncio
    async def test_agent_endpoint_full_pipeline(self, test_users):
        """Test the agent endpoint with real RAG generation."""
        # Setup environment for user1
        user1_env = self.setup_user_environment(test_users[0].id)
        
        # Mock LLM response
        async def mock_llm_generate(*args, **kwargs):
            return {
                "content": "Based on the retrieved information about RAG, it combines retrieval and generation for better results.",
                "usage": {"total_tokens": 50}
            }
        
        with patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user') as mock_auth:
            mock_auth.return_value = test_users[0]
            
            # Create actual database instances
            media_db = MediaDatabase(str(user1_env["media_db"]), client_id=str(test_users[0].id))
            chacha_db = CharactersRAGDB(str(user1_env["chacha_db"]), client_id=str(test_users[0].id))
            
            with patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user') as mock_media_db:
                with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user') as mock_chacha_db:
                    mock_media_db.return_value = media_db
                    mock_chacha_db.return_value = chacha_db
                    
                    # Mock the RAG service's generate_answer method
                    with patch('tldw_Server_API.app.core.RAG.rag_service.integration.RAGService.generate_answer') as mock_generate:
                        mock_generate.return_value = mock_llm_generate()
                        
                        transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://test") as client:
                            # Test 1: Basic RAG generation
                            response = await client.post(
                                "/api/v1/retrieval_agent/agent",
                                json={
                                    "message": {
                                        "role": "user",
                                        "content": "What is RAG and how does it work?"
                                    },
                                    "mode": "rag",
                                    "search_mode": "custom",
                                    "rag_generation_config": {
                                        "model": "test-model",
                                        "temperature": 0.7,
                                        "max_tokens_to_sample": 500
                                    }
                                },
                                headers={"X-API-KEY": "default-secret-key-for-single-user"}
                            )
                            
                            assert response.status_code == 200
                            data = response.json()
                            assert "response_message" in data
                            assert data["response_message"]["role"] == "assistant"
                            assert len(data["response_message"]["content"]) > 0
                            assert "conversation_id" in data
                            
                            # Test 2: Research mode
                            response = await client.post(
                                "/api/v1/retrieval_agent/agent",
                                json={
                                    "message": {
                                        "role": "user",
                                        "content": "Research the benefits of using vector databases in RAG systems"
                                    },
                                    "mode": "research",
                                    "search_settings": {
                                        "search_databases": ["media_db", "notes"],
                                        "limit": 10
                                    }
                                },
                                headers={"X-API-KEY": "default-secret-key-for-single-user"}
                            )
                            
                            assert response.status_code == 200
                            data = response.json()
                            assert "response_message" in data
    
    @pytest.mark.asyncio
    async def test_multi_user_isolation(self, test_users):
        """Test that multiple users have isolated RAG services."""
        # Setup environments for both users
        user1_env = self.setup_user_environment(test_users[0].id)
        user2_env = self.setup_user_environment(test_users[1].id)
        
        # Add unique data to user2's database
        user2_media_db = MediaDatabase(str(user2_env["media_db"]), client_id=str(test_users[1].id))
        user2_media_db.add_media_with_keywords(
            url="https://example.com/user2-doc",
            title="User2 Specific Document",
            media_type="document",
            content="This document belongs only to user2 and should not appear in user1 searches.",
            keywords=["user2", "specific", "private"],
            prompt="",
            transcription_model="test-model",
            author="User2",
            ingestion_date=datetime.now().isoformat()
        )
        
        # Mock the RAG service for user1
        from tldw_Server_API.app.api.v1.endpoints.rag import get_rag_service_for_user
        async def mock_get_rag_service_user1():
            rag_service = RAGService(
                config=RAGConfig(),
                media_db_path=Path(user1_env["media_db"]),
                chachanotes_db_path=Path(user1_env["chacha_db"]),
                chroma_path=Path(user1_env["chroma_dir"]),
                llm_handler=None
            )
            await rag_service.initialize()
            return rag_service
        
        # Mock the RAG service for user2
        async def mock_get_rag_service_user2():
            rag_service = RAGService(
                config=RAGConfig(),
                media_db_path=Path(user2_env["media_db"]),
                chachanotes_db_path=Path(user2_env["chacha_db"]),
                chroma_path=Path(user2_env["chroma_dir"]),
                llm_handler=None
            )
            await rag_service.initialize()
            return rag_service
        
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Search as user1
            with patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user') as mock_auth:
                mock_auth.return_value = test_users[0]
                
                # Create actual database instances for user1
                media_db1 = MediaDatabase(str(user1_env["media_db"]), client_id=str(test_users[0].id))
                chacha_db1 = CharactersRAGDB(str(user1_env["chacha_db"]), client_id=str(test_users[0].id))
                
                with patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user') as mock_media_db:
                    with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user') as mock_chacha_db:
                        mock_media_db.return_value = media_db1
                        mock_chacha_db.return_value = chacha_db1
                        
                        # Override RAG service for user1
                        app.dependency_overrides[get_rag_service_for_user] = mock_get_rag_service_user1
                        
                        try:
                            response = await client.post(
                                "/api/v1/retrieval_agent/search",
                                json={
                                    "querystring": "User2 Specific Document",
                                    "search_mode": "basic"
                                },
                                headers={"X-API-KEY": "default-secret-key-for-single-user"}
                            )
                            
                            assert response.status_code == 200
                            data = response.json()
                            # User1 should not find user2's document
                            assert not any("User2 Specific" in result["title"] for result in data["results"])
                        finally:
                            if get_rag_service_for_user in app.dependency_overrides:
                                del app.dependency_overrides[get_rag_service_for_user]
            
            # Search as user2
            with patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user') as mock_auth:
                mock_auth.return_value = test_users[1]
                
                # Create actual database instances for user2
                media_db2 = MediaDatabase(str(user2_env["media_db"]), client_id=str(test_users[1].id))
                chacha_db2 = CharactersRAGDB(str(user2_env["chacha_db"]), client_id=str(test_users[1].id))
                
                with patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user') as mock_media_db:
                    with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user') as mock_chacha_db:
                        mock_media_db.return_value = media_db2
                        mock_chacha_db.return_value = chacha_db2
                        
                        # Override RAG service for user2
                        app.dependency_overrides[get_rag_service_for_user] = mock_get_rag_service_user2
                        
                        try:
                            response = await client.post(
                                "/api/v1/retrieval_agent/search",
                                json={
                                    "querystring": "User2 Specific Document",
                                    "search_mode": "basic"
                                },
                                headers={"X-API-KEY": "default-secret-key-for-single-user"}
                            )
                            
                            assert response.status_code == 200
                            data = response.json()
                            # User2 should find their own document
                            assert any("User2 Specific" in result["title"] for result in data["results"])
                        finally:
                            if get_rag_service_for_user in app.dependency_overrides:
                                del app.dependency_overrides[get_rag_service_for_user]
    
    @pytest.mark.asyncio
    async def test_streaming_response(self, test_users):
        """Test streaming response functionality."""
        # Setup environment
        user1_env = self.setup_user_environment(test_users[0].id)
        
        # Mock streaming LLM response
        async def mock_llm_stream(*args, **kwargs):
            chunks = [
                {"type": "content", "content": "Based on "},
                {"type": "content", "content": "the retrieved information, "},
                {"type": "content", "content": "RAG combines retrieval and generation."},
                {"type": "citation", "citation": {"source_name": "Introduction to RAG", "content": "RAG is a powerful technique"}}
            ]
            for chunk in chunks:
                yield chunk
        
        # Mock the RAG service
        from tldw_Server_API.app.api.v1.endpoints.rag import get_rag_service_for_user
        async def mock_get_rag_service():
            rag_service = RAGService(
                config=RAGConfig(),
                media_db_path=Path(user1_env["media_db"]),
                chachanotes_db_path=Path(user1_env["chacha_db"]),
                chroma_path=Path(user1_env["chroma_dir"]),
                llm_handler=None
            )
            await rag_service.initialize()
            # Mock the streaming method
            rag_service.generate_answer_stream = mock_llm_stream
            return rag_service
        
        with patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user') as mock_auth:
            mock_auth.return_value = test_users[0]
            
            # Create actual database instances
            media_db = MediaDatabase(str(user1_env["media_db"]), client_id=str(test_users[0].id))
            chacha_db = CharactersRAGDB(str(user1_env["chacha_db"]), client_id=str(test_users[0].id))
            
            with patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user') as mock_media_db:
                with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user') as mock_chacha_db:
                    mock_media_db.return_value = media_db
                    mock_chacha_db.return_value = chacha_db
                    
                    # Override the RAG service creation
                    app.dependency_overrides[get_rag_service_for_user] = mock_get_rag_service
                    
                    try:
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            response = await client.post(
                                "/api/v1/retrieval_agent/agent",
                                json={
                                    "message": {
                                        "role": "user",
                                        "content": "What is RAG?"
                                    },
                                    "mode": "rag",
                                    "rag_generation_config": {
                                        "stream": True,
                                        "model": "test-model"
                                    }
                                },
                                headers={"X-API-KEY": "default-secret-key-for-single-user"}
                            )
                            
                            assert response.status_code == 200
                            assert "text/event-stream" in response.headers["content-type"]
                            
                            # Parse SSE events
                            events = []
                            for line in response.iter_lines():
                                if line.startswith("data: "):
                                    events.append(json.loads(line[6:]))
                            
                            # Verify we got all event types
                            assert any(e["type"] == "start" for e in events)
                            assert any(e["type"] == "content" for e in events)
                            assert any(e["type"] == "citation" for e in events)
                            assert any(e["type"] == "end" for e in events)
                    finally:
                        if get_rag_service_for_user in app.dependency_overrides:
                            del app.dependency_overrides[get_rag_service_for_user]
    
    @pytest.mark.asyncio
    async def test_error_handling(self, test_users):
        """Test error handling throughout the pipeline."""
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            with patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user') as mock_auth:
                mock_auth.return_value = test_users[0]
                
                # Test 1: Empty message
                response = await client.post(
                    "/api/v1/retrieval_agent/agent",
                    json={
                        "message": {
                            "role": "user",
                            "content": ""
                        }
                    },
                    headers={"X-API-KEY": "default-secret-key-for-single-user"}
                )
                assert response.status_code == 422
                
                # Test 2: Invalid search mode
                response = await client.post(
                    "/api/v1/retrieval_agent/search",
                    json={
                        "querystring": "test",
                        "search_mode": "invalid_mode"
                    },
                    headers={"X-API-KEY": "default-secret-key-for-single-user"}
                )
                assert response.status_code == 422
                
                # Test 3: Database error simulation
                from tldw_Server_API.app.api.v1.endpoints.rag import get_rag_service_for_user
                
                async def override_get_rag_service():
                    from fastapi import HTTPException, status
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Database connection failed"
                    )
                
                app.dependency_overrides[get_rag_service_for_user] = override_get_rag_service
                try:
                    response = await client.post(
                        "/api/v1/retrieval_agent/search",
                        json={
                            "querystring": "test",
                            "search_mode": "basic"
                        },
                        headers={"X-API-KEY": "default-secret-key-for-single-user"}
                    )
                    assert response.status_code == 500
                finally:
                    # Clean up override
                    del app.dependency_overrides[get_rag_service_for_user]
    
    @pytest.mark.asyncio
    async def test_conversation_context(self, test_users):
        """Test conversation context handling."""
        user1_env = self.setup_user_environment(test_users[0].id)
        
        with patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user') as mock_auth:
            mock_auth.return_value = test_users[0]
            
            # Create actual database instances
            media_db = MediaDatabase(str(user1_env["media_db"]), client_id=str(test_users[0].id))
            chacha_db = CharactersRAGDB(str(user1_env["chacha_db"]), client_id=str(test_users[0].id))
            
            with patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user') as mock_media_db:
                with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user') as mock_chacha_db:
                    mock_media_db.return_value = media_db
                    mock_chacha_db.return_value = chacha_db
                    
                    # Mock LLM to check conversation history
                    conversation_history_checked = False
                    
                    async def mock_llm_generate(*args, **kwargs):
                        nonlocal conversation_history_checked
                        if "conversation_history" in kwargs and kwargs["conversation_history"]:
                            conversation_history_checked = True
                        return {"content": "Response with context", "usage": {"total_tokens": 50}}
                    
                    with patch('tldw_Server_API.app.core.RAG.rag_service.integration.RAGService.generate_answer') as mock_generate:
                        async def async_mock_llm_generate(*args, **kwargs):
                            return await mock_llm_generate(*args, **kwargs)
                        mock_generate.side_effect = async_mock_llm_generate
                        
                        transport = ASGITransport(app=app)
                    async with AsyncClient(transport=transport, base_url="http://test") as client:
                            # First message
                            response1 = await client.post(
                                "/api/v1/retrieval_agent/agent",
                                json={
                                    "message": {
                                        "role": "user",
                                        "content": "Tell me about RAG"
                                    },
                                    "mode": "rag"
                                },
                                headers={"X-API-KEY": "default-secret-key-for-single-user"}
                            )
                            
                            assert response1.status_code == 200
                            data1 = response1.json()
                            conversation_id = data1["conversation_id"]
                            
                            # Second message with conversation ID
                            response2 = await client.post(
                                "/api/v1/retrieval_agent/agent",
                                json={
                                    "message": {
                                        "role": "user",
                                        "content": "What are its benefits?"
                                    },
                                    "conversation_id": conversation_id,
                                    "mode": "rag"
                                },
                                headers={"X-API-KEY": "default-secret-key-for-single-user"}
                            )
                            
                            assert response2.status_code == 200
                            data2 = response2.json()
                            assert data2["conversation_id"] == conversation_id


class TestRAGPerformance:
    """Performance tests for the RAG pipeline."""
    
    @pytest.fixture
    def large_dataset_env(self):
        """Create a large dataset for performance testing."""
        temp_dir = tempfile.mkdtemp()
        user_dir = Path(temp_dir) / "1"
        user_dir.mkdir(exist_ok=True)
        db_path = user_dir / "user_media_library.sqlite"
        
        # Create database using MediaDatabase to ensure proper schema
        media_db = MediaDatabase(str(db_path), client_id="1")
        
        # Add 1000 test entries
        for i in range(1000):
            media_db.add_media_with_keywords(
                url=f"https://example.com/doc-{i}",
                title=f"Document {i}: {['RAG', 'ML', 'AI', 'Python', 'Data'][i % 5]} Guide",
                media_type=["article", "video", "document"][i % 3],
                content=f"This is test content for document {i}. It contains information about {['retrieval', 'machine learning', 'artificial intelligence', 'programming', 'data science'][i % 5]}.",
                keywords=[['RAG', 'ML', 'AI', 'Python', 'Data'][i % 5].lower()],
                prompt="",
                transcription_model="test-model",
                author=f"Author {i % 10}",
                ingestion_date=datetime.now().isoformat()
            )
        
        yield {
            "temp_dir": temp_dir,
            "db_path": db_path,
            "user_dir": user_dir
        }
        
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    @pytest.mark.asyncio
    async def test_search_performance(self, large_dataset_env):
        """Test search performance with large dataset."""
        import time
        
        user = User(id=1, username="perftest", email="perf@test.com", is_active=True)
        
        # Mock the RAG service
        from tldw_Server_API.app.api.v1.endpoints.rag import get_rag_service_for_user
        async def mock_get_rag_service():
            rag_service = RAGService(
                config=RAGConfig(),
                media_db_path=Path(large_dataset_env["db_path"]),
                chachanotes_db_path=Path(large_dataset_env["user_dir"]) / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite",
                chroma_path=Path(large_dataset_env["user_dir"]) / "chroma",
                llm_handler=None
            )
            await rag_service.initialize()
            return rag_service
        
        with patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user') as mock_auth:
            mock_auth.return_value = user
            
            # Create actual database instances
            media_db = MediaDatabase(str(large_dataset_env["db_path"]), client_id=str(user.id))
            chacha_db_path = large_dataset_env["user_dir"] / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite"
            chacha_db_path.parent.mkdir(parents=True, exist_ok=True)
            chacha_db = CharactersRAGDB(str(chacha_db_path), client_id=str(user.id))
            
            with patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user') as mock_media_db:
                with patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user') as mock_chacha_db:
                    # Temporarily set the user database directory
                    original_user_db_dir = settings.get("USER_DB_BASE_DIR", "/tmp/users")
                    settings["USER_DB_BASE_DIR"] = str(large_dataset_env["temp_dir"])
                    try:
                        mock_media_db.return_value = media_db
                        mock_chacha_db.return_value = chacha_db
                        
                        # Override the RAG service creation
                        app.dependency_overrides[get_rag_service_for_user] = mock_get_rag_service
                        
                        transport = ASGITransport(app=app)
                        async with AsyncClient(transport=transport, base_url="http://test") as client:
                            # Measure search time
                            start_time = time.time()
                            
                            response = await client.post(
                                "/api/v1/retrieval_agent/search",
                                json={
                                    "querystring": "machine learning",
                                    "search_mode": "basic",
                                    "limit": 50
                                },
                                headers={"X-API-KEY": "default-secret-key-for-single-user"}
                            )
                            
                            end_time = time.time()
                            search_time = end_time - start_time
                            
                            assert response.status_code == 200
                            data = response.json()
                            assert len(data["results"]) > 0
                            
                            # Performance assertion - search should complete within 2 seconds
                            assert search_time < 2.0, f"Search took {search_time:.2f} seconds, expected < 2.0s"
                            
                            print(f"Search performance: {search_time:.2f}s for {len(data['results'])} results")
                    finally:
                        settings["USER_DB_BASE_DIR"] = original_user_db_dir
                        if get_rag_service_for_user in app.dependency_overrides:
                            del app.dependency_overrides[get_rag_service_for_user]