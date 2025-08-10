# test_rag_v2_endpoints.py
# Tests for the new simplified RAG v2 endpoints
import json
import tempfile
import shutil
import sqlite3
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, AsyncMock, MagicMock
import pytest
from httpx import AsyncClient
from fastapi import FastAPI
from fastapi.testclient import TestClient

from tldw_Server_API.app.main import app
from tldw_Server_API.app.api.v1.schemas.rag_schemas_simple import (
    SimpleSearchRequest,
    SimpleSearchResponse,
    SearchResult,
    AdvancedSearchRequest,
    AdvancedSearchResponse,
    SimpleAgentRequest,
    SimpleAgentResponse,
    Source,
    AdvancedAgentRequest,
    AdvancedAgentResponse,
    SearchType,
    AgentMode,
    SearchStrategy,
    HybridSearchConfig,
    SemanticSearchConfig,
    SearchConfig,
    GenerationConfig,
    AgentSearchConfig,
    ResearchTool
)
from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import User
from tldw_Server_API.app.core.config import settings


class TestRAGV2Endpoints:
    """Test suite for new RAG v2 endpoints"""
    
    @pytest.fixture(autouse=True)
    def setup_and_teardown(self):
        """Setup test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.user_base_dir = Path(self.temp_dir) / "users"
        self.user_base_dir.mkdir(exist_ok=True)
        
        # Store original settings
        self.original_user_db_dir = settings.get("USER_DB_BASE_DIR", "/tmp/users")
        settings["USER_DB_BASE_DIR"] = str(self.user_base_dir)
        
        # Create test client
        self.client = TestClient(app)
        
        # Default API key for authentication
        self.DEFAULT_API_KEY = "default-secret-key-for-single-user"
        self.auth_headers = {"Authorization": f"Bearer {self.DEFAULT_API_KEY}"}
        
        # Mock user for authentication
        self.test_user = User(id=1, username="testuser", email="test@example.com", is_active=True)
        
        yield
        
        # Cleanup
        settings["USER_DB_BASE_DIR"] = self.original_user_db_dir
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def create_test_database(self, user_id: int):
        """Create test databases for a user"""
        user_dir = self.user_base_dir / str(user_id)
        user_dir.mkdir(exist_ok=True)
        
        # Create media database
        media_db = user_dir / "user_media_library.sqlite"
        conn = sqlite3.connect(str(media_db))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS Media (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT UNIQUE,
                title TEXT NOT NULL,
                type TEXT NOT NULL,
                content TEXT,
                author TEXT,
                ingestion_date DATETIME,
                transcription_model TEXT,
                is_trash BOOLEAN DEFAULT 0 NOT NULL,
                content_hash TEXT UNIQUE NOT NULL,
                uuid TEXT UNIQUE NOT NULL,
                last_modified DATETIME NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                client_id TEXT NOT NULL,
                deleted BOOLEAN NOT NULL DEFAULT 0
            )
        """)
        
        # Add test data
        import hashlib
        test_data = []
        for i, (title, content, url) in enumerate([
            ("RAG Overview", "Retrieval-Augmented Generation combines retrieval with generation.", "http://example.com/rag"),
            ("Python Tutorial", "Python is a versatile programming language.", "http://example.com/python"),
            ("Machine Learning", "ML enables computers to learn from data.", "http://example.com/ml")
        ]):
            content_hash = hashlib.md5(content.encode()).hexdigest()
            uuid_val = f"test_uuid_{i}"
            test_data.append((
                url, title, "article", content, "Test Author", datetime.now().isoformat(),
                "test", content_hash, uuid_val, datetime.now().isoformat(), 
                1, "test_client", 0
            ))
        
        cursor.executemany(
            """INSERT INTO Media (url, title, type, content, author, ingestion_date,
                                transcription_model, content_hash, uuid, last_modified,
                                version, client_id, deleted)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            test_data
        )
        
        conn.commit()
        conn.close()
        
        # Create ChaChaNotes database
        chacha_dir = user_dir / "chachanotes_user_dbs"
        chacha_dir.mkdir(exist_ok=True)
        chacha_db = chacha_dir / "user_chacha_notes_rag.sqlite"
        
        conn = sqlite3.connect(str(chacha_db))
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS notes (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                last_modified DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
                deleted BOOLEAN NOT NULL DEFAULT 0,
                client_id TEXT NOT NULL DEFAULT 'unknown',
                version INTEGER NOT NULL DEFAULT 1
            )
        """)
        
        # Add test notes
        cursor.execute(
            """INSERT INTO notes (id, title, content, created_at, last_modified, client_id)
               VALUES (?, ?, ?, ?, ?, ?)""",
            ("note1", "RAG Notes", "Important RAG information", datetime.now().isoformat(), 
             datetime.now().isoformat(), "test_client")
        )
        
        conn.commit()
        conn.close()
        
        # Create chroma directory
        chroma_dir = user_dir / "chroma"
        chroma_dir.mkdir(exist_ok=True)
        
        return {
            "media_db": media_db,
            "chacha_db": chacha_db,
            "chroma_dir": chroma_dir
        }
    
    # ============= Simple Search Tests =============
    
    def test_simple_search_basic(self):
        """Test simple search endpoint with basic query"""
        from fastapi import Depends
        from tldw_Server_API.app.main import app
        
        # Create test database
        self.create_test_database(self.test_user.id)
        
        # Override dependencies for testing
        async def override_get_user():
            return self.test_user
        
        async def override_get_media_db():
            mock_db = Mock()
            mock_db.db_path = str(self.user_base_dir / "1" / "user_media_library.sqlite")
            return mock_db
        
        async def override_get_chacha_db():
            mock_db = Mock()
            mock_db.db_path = str(self.user_base_dir / "1" / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite")
            return mock_db
        
        # Mock RAG service
        async def override_get_rag_service():
            mock_service = AsyncMock()
            mock_service.search = AsyncMock(return_value=[
                {
                    "id": "1",
                    "title": "RAG Overview",
                    "content": "Retrieval-Augmented Generation combines retrieval with generation for better AI responses.",
                    "score": 0.95,
                    "source": "media_db",
                    "metadata": {"author": "AI Expert"}
                }
            ])
            mock_service.initialize = AsyncMock()
            mock_service.cleanup = Mock()
            return mock_service
        
        # Apply overrides
        from tldw_Server_API.app.core.AuthNZ.User_DB_Handling import get_request_user
        from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import get_media_db_for_user
        from tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps import get_chacha_db_for_user
        from tldw_Server_API.app.api.v1.endpoints.rag_v2 import get_rag_service
        
        app.dependency_overrides[get_request_user] = override_get_user
        app.dependency_overrides[get_media_db_for_user] = override_get_media_db
        app.dependency_overrides[get_chacha_db_for_user] = override_get_chacha_db
        app.dependency_overrides[get_rag_service] = override_get_rag_service
        
        mock_media_db.return_value = Mock(db_path=str(self.user_base_dir / "1" / "user_media_library.sqlite"))
        mock_chacha_db.return_value = Mock(db_path=str(self.user_base_dir / "1" / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite"))
        
        # Mock RAG service
        mock_service_instance = AsyncMock()
        mock_service_instance.search = AsyncMock(return_value=[
            {
                "id": "1",
                "title": "RAG Overview",
                "content": "Retrieval-Augmented Generation combines retrieval with generation for better AI responses.",
                "score": 0.95,
                "source": "media_db",
                "metadata": {"author": "AI Expert"}
            }
        ])
        mock_service_instance.initialize = AsyncMock()
        mock_service_instance.cleanup = Mock()
        mock_rag_service.return_value = mock_service_instance
        
        # Make request
        response = self.client.post(
            "/api/v1/rag/search",
            headers=self.auth_headers,
            json={
                "query": "What is RAG?",
                "search_type": "hybrid",
                "limit": 10,
                "databases": ["media_db"],
                "keywords": ["AI", "RAG"]
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert len(data["results"]) == 1
        assert data["results"][0]["title"] == "RAG Overview"
        assert data["search_type_used"] == "hybrid"
        assert "query_id" in data
    
    @patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user')
    @patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user')
    @patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user')
    @patch('tldw_Server_API.app.api.v1.endpoints.rag_v2.RAGService')
    def test_simple_search_semantic(self, mock_rag_service, mock_chacha_db, mock_media_db, mock_user):
        """Test simple search with semantic search type"""
        mock_user.return_value = self.test_user
        self.create_test_database(self.test_user.id)
        
        mock_media_db.return_value = Mock(db_path=str(self.user_base_dir / "1" / "user_media_library.sqlite"))
        mock_chacha_db.return_value = Mock(db_path=str(self.user_base_dir / "1" / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite"))
        
        mock_service_instance = AsyncMock()
        mock_service_instance.search = AsyncMock(return_value=[])
        mock_service_instance.initialize = AsyncMock()
        mock_service_instance.cleanup = Mock()
        mock_rag_service.return_value = mock_service_instance
        
        response = self.client.post(
            "/api/v1/rag/search",
            headers=self.auth_headers,
            json={
                "query": "machine learning concepts",
                "search_type": "semantic",
                "limit": 5
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["search_type_used"] == "semantic"
        
        # Verify semantic search was configured correctly
        mock_service_instance.search.assert_called_once()
        call_args = mock_service_instance.search.call_args
        assert call_args[1]["use_semantic_search"] == True
        assert call_args[1]["use_fulltext_search"] == False
    
    @patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user')
    def test_simple_search_invalid_database(self, mock_user):
        """Test simple search with invalid database name"""
        mock_user.return_value = self.test_user
        
        response = self.client.post(
            "/api/v1/rag/search",
            headers=self.auth_headers,
            json={
                "query": "test",
                "databases": ["invalid_db"]
            }
        )
        
        assert response.status_code == 422  # Validation error
    
    # ============= Advanced Search Tests =============
    
    @patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user')
    @patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user')
    @patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user')
    @patch('tldw_Server_API.app.api.v1.endpoints.rag_v2.RAGService')
    def test_advanced_search_with_filters(self, mock_rag_service, mock_chacha_db, mock_media_db, mock_user):
        """Test advanced search with complex filters"""
        mock_user.return_value = self.test_user
        self.create_test_database(self.test_user.id)
        
        mock_media_db.return_value = Mock(db_path=str(self.user_base_dir / "1" / "user_media_library.sqlite"))
        mock_chacha_db.return_value = Mock(db_path=str(self.user_base_dir / "1" / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite"))
        
        mock_service_instance = AsyncMock()
        mock_service_instance.search = AsyncMock(return_value=[
            {
                "id": "1",
                "title": "Advanced Result",
                "content": "This is an advanced search result with filters applied.",
                "score": 0.88,
                "source": "media_db",
                "metadata": {"date": "2024-01-15", "author": "Expert"}
            }
        ])
        mock_service_instance.initialize = AsyncMock()
        mock_service_instance.cleanup = Mock()
        mock_rag_service.return_value = mock_service_instance
        
        response = self.client.post(
            "/api/v1/rag/search/advanced",
            headers=self.auth_headers,
            json={
                "query": "advanced query",
                "search_config": {
                    "search_type": "hybrid",
                    "limit": 20,
                    "offset": 10,
                    "databases": ["media_db", "notes"],
                    "keywords": ["AI", "ML"],
                    "date_range": {
                        "start": "2024-01-01",
                        "end": "2024-12-31"
                    },
                    "metadata_filters": {
                        "author": "Expert"
                    },
                    "include_scores": True,
                    "include_full_content": True
                },
                "hybrid_config": {
                    "semantic_weight": 7.0,
                    "fulltext_weight": 3.0,
                    "rrf_k": 60
                },
                "strategy": "query_fusion"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["strategy_used"] == "query_fusion"
        assert "search_config" in data
        
        # Verify advanced configuration was passed
        mock_service_instance.search.assert_called_once()
        call_args = mock_service_instance.search.call_args
        assert "filters" in call_args[1]
        assert call_args[1]["search_strategy"] == "query_fusion"
    
    @patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user')
    @patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user')
    @patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user')
    @patch('tldw_Server_API.app.api.v1.endpoints.rag_v2.RAGService')
    def test_advanced_search_hyde_strategy(self, mock_rag_service, mock_chacha_db, mock_media_db, mock_user):
        """Test advanced search with HYDE strategy"""
        mock_user.return_value = self.test_user
        self.create_test_database(self.test_user.id)
        
        mock_media_db.return_value = Mock(db_path=str(self.user_base_dir / "1" / "user_media_library.sqlite"))
        mock_chacha_db.return_value = Mock(db_path=str(self.user_base_dir / "1" / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite"))
        
        mock_service_instance = AsyncMock()
        mock_service_instance.search = AsyncMock(return_value=[])
        mock_service_instance.initialize = AsyncMock()
        mock_service_instance.cleanup = Mock()
        mock_rag_service.return_value = mock_service_instance
        
        response = self.client.post(
            "/api/v1/rag/search/advanced",
            headers=self.auth_headers,
            json={
                "query": "explain quantum computing",
                "strategy": "hyde",
                "semantic_config": {
                    "similarity_threshold": 0.7,
                    "rerank": True
                }
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["strategy_used"] == "hyde"
    
    # ============= Simple Agent Tests =============
    
    @patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user')
    @patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user')
    @patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user')
    @patch('tldw_Server_API.app.api.v1.endpoints.rag_v2.RAGService')
    def test_simple_agent_qa(self, mock_rag_service, mock_chacha_db, mock_media_db, mock_user):
        """Test simple agent Q&A functionality"""
        mock_user.return_value = self.test_user
        self.create_test_database(self.test_user.id)
        
        mock_media_db.return_value = Mock(db_path=str(self.user_base_dir / "1" / "user_media_library.sqlite"))
        
        # Mock chacha_db with conversation methods
        mock_chacha_instance = Mock()
        mock_chacha_instance.db_path = str(self.user_base_dir / "1" / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite")
        mock_chacha_instance.get_messages_for_conversation = Mock(return_value=[])
        mock_chacha_instance.save_message = Mock()
        mock_chacha_db.return_value = mock_chacha_instance
        
        # Mock RAG service
        mock_service_instance = AsyncMock()
        mock_service_instance.search = AsyncMock(return_value=[
            {
                "id": "1",
                "title": "RAG Overview",
                "content": "RAG combines retrieval and generation.",
                "score": 0.9,
                "source": "media_db"
            }
        ])
        mock_service_instance.generate = AsyncMock(
            return_value="RAG is a technique that combines information retrieval with text generation."
        )
        mock_service_instance.initialize = AsyncMock()
        mock_service_instance.cleanup = Mock()
        mock_rag_service.return_value = mock_service_instance
        
        response = self.client.post(
            "/api/v1/rag/agent",
            headers=self.auth_headers,
            json={
                "message": "What is RAG?",
                "search_databases": ["media_db"],
                "model": "gpt-4"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "response" in data
        assert "conversation_id" in data
        assert "sources" in data
        assert len(data["sources"]) > 0
        assert data["sources"][0]["title"] == "RAG Overview"
    
    @patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user')
    @patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user')
    @patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user')
    @patch('tldw_Server_API.app.api.v1.endpoints.rag_v2.RAGService')
    def test_simple_agent_with_conversation(self, mock_rag_service, mock_chacha_db, mock_media_db, mock_user):
        """Test simple agent with existing conversation context"""
        mock_user.return_value = self.test_user
        self.create_test_database(self.test_user.id)
        
        mock_media_db.return_value = Mock(db_path=str(self.user_base_dir / "1" / "user_media_library.sqlite"))
        
        # Mock chacha_db with conversation history
        mock_chacha_instance = Mock()
        mock_chacha_instance.db_path = str(self.user_base_dir / "1" / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite")
        mock_chacha_instance.get_messages_for_conversation = Mock(return_value=[
            {"sender": "user", "content": "Previous question"},
            {"sender": "assistant", "content": "Previous answer"}
        ])
        mock_chacha_instance.save_message = Mock()
        mock_chacha_db.return_value = mock_chacha_instance
        
        mock_service_instance = AsyncMock()
        mock_service_instance.search = AsyncMock(return_value=[])
        mock_service_instance.generate = AsyncMock(return_value="Follow-up answer based on context.")
        mock_service_instance.initialize = AsyncMock()
        mock_service_instance.cleanup = Mock()
        mock_rag_service.return_value = mock_service_instance
        
        response = self.client.post(
            "/api/v1/rag/agent",
            headers=self.auth_headers,
            json={
                "message": "Tell me more",
                "conversation_id": "existing_conv_123"
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["conversation_id"] == "existing_conv_123"
        
        # Verify conversation history was loaded
        mock_chacha_instance.get_messages_for_conversation.assert_called_once_with(
            "existing_conv_123",
            limit=20,
            order_by_timestamp="ASC"
        )
    
    # ============= Advanced Agent Tests =============
    
    @patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user')
    @patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user')
    @patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user')
    @patch('tldw_Server_API.app.api.v1.endpoints.rag_v2.RAGService')
    def test_advanced_agent_research_mode(self, mock_rag_service, mock_chacha_db, mock_media_db, mock_user):
        """Test advanced agent in research mode"""
        mock_user.return_value = self.test_user
        self.create_test_database(self.test_user.id)
        
        mock_media_db.return_value = Mock(db_path=str(self.user_base_dir / "1" / "user_media_library.sqlite"))
        
        mock_chacha_instance = Mock()
        mock_chacha_instance.db_path = str(self.user_base_dir / "1" / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite")
        mock_chacha_instance.get_messages_for_conversation = Mock(return_value=[])
        mock_chacha_instance.save_message = Mock()
        mock_chacha_db.return_value = mock_chacha_instance
        
        mock_service_instance = AsyncMock()
        mock_service_instance.search = AsyncMock(return_value=[
            {
                "id": "1",
                "title": "Research Result",
                "content": "Detailed research content",
                "score": 0.95,
                "source": "media_db"
            }
        ])
        mock_service_instance.generate = AsyncMock(
            return_value="Comprehensive research answer with multiple perspectives."
        )
        mock_service_instance.initialize = AsyncMock()
        mock_service_instance.cleanup = Mock()
        mock_rag_service.return_value = mock_service_instance
        
        response = self.client.post(
            "/api/v1/rag/agent/advanced",
            headers=self.auth_headers,
            json={
                "message": "Research quantum computing applications",
                "mode": "research",
                "generation_config": {
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "max_tokens": 2048,
                    "stream": False
                },
                "search_config": {
                    "search_type": "hybrid",
                    "databases": ["media_db", "notes"],
                    "keywords": ["quantum", "computing"],
                    "limit": 15
                },
                "tools": ["web_search", "reasoning"],
                "system_prompt": "You are a research assistant specializing in quantum computing."
            }
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["mode_used"] == "research"
        assert "tools_used" in data
        assert "search_stats" in data
        assert "generation_stats" in data
        assert data["generation_stats"]["model"] == "gpt-4"
    
    @patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user')
    @patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user')
    @patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user')
    @patch('tldw_Server_API.app.api.v1.endpoints.rag_v2.RAGService')
    def test_advanced_agent_streaming(self, mock_rag_service, mock_chacha_db, mock_media_db, mock_user):
        """Test advanced agent with streaming response"""
        mock_user.return_value = self.test_user
        self.create_test_database(self.test_user.id)
        
        mock_media_db.return_value = Mock(db_path=str(self.user_base_dir / "1" / "user_media_library.sqlite"))
        
        mock_chacha_instance = Mock()
        mock_chacha_instance.db_path = str(self.user_base_dir / "1" / "chachanotes_user_dbs" / "user_chacha_notes_rag.sqlite")
        mock_chacha_instance.get_messages_for_conversation = Mock(return_value=[])
        mock_chacha_db.return_value = mock_chacha_instance
        
        # Mock streaming response
        async def mock_stream(*args, **kwargs):
            yield "First "
            yield "chunk "
            yield "of response."
        
        mock_service_instance = AsyncMock()
        mock_service_instance.search = AsyncMock(return_value=[])
        mock_service_instance.generate_stream = mock_stream
        mock_service_instance.initialize = AsyncMock()
        mock_service_instance.cleanup = Mock()
        mock_rag_service.return_value = mock_service_instance
        
        response = self.client.post(
            "/api/v1/rag/agent/advanced",
            headers=self.auth_headers,
            json={
                "message": "Stream this response",
                "generation_config": {
                    "stream": True
                }
            }
        )
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "text/event-stream; charset=utf-8"
    
    # ============= Error Handling Tests =============
    
    @patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user')
    @patch('tldw_Server_API.app.api.v1.API_Deps.DB_Deps.get_media_db_for_user')
    @patch('tldw_Server_API.app.api.v1.API_Deps.ChaCha_Notes_DB_Deps.get_chacha_db_for_user')
    @patch('tldw_Server_API.app.api.v1.endpoints.rag_v2.RAGService')
    def test_search_error_handling(self, mock_rag_service, mock_chacha_db, mock_media_db, mock_user):
        """Test error handling in search endpoint"""
        mock_user.return_value = self.test_user
        
        mock_media_db.return_value = Mock(db_path="/test/path")
        mock_chacha_db.return_value = Mock(db_path="/test/path")
        
        # Mock service to raise an error
        mock_service_instance = AsyncMock()
        mock_service_instance.search = AsyncMock(side_effect=Exception("Database connection failed"))
        mock_service_instance.initialize = AsyncMock()
        mock_service_instance.cleanup = Mock()
        mock_rag_service.return_value = mock_service_instance
        
        response = self.client.post(
            "/api/v1/rag/search",
            headers=self.auth_headers,
            json={
                "query": "test query"
            }
        )
        
        assert response.status_code == 500
        data = response.json()
        assert "detail" in data
        assert "Search operation failed" in data["detail"]
    
    @patch('tldw_Server_API.app.core.AuthNZ.User_DB_Handling.get_request_user')
    def test_invalid_search_request(self, mock_user):
        """Test validation errors for invalid requests"""
        mock_user.return_value = self.test_user
        
        # Empty query
        response = self.client.post(
            "/api/v1/rag/search",
            headers=self.auth_headers,
            json={
                "query": ""
            }
        )
        assert response.status_code == 422
        
        # Invalid limit
        response = self.client.post(
            "/api/v1/rag/search",
            headers=self.auth_headers,
            json={
                "query": "test",
                "limit": 0
            }
        )
        assert response.status_code == 422
        
        # Limit too high
        response = self.client.post(
            "/api/v1/rag/search",
            headers=self.auth_headers,
            json={
                "query": "test",
                "limit": 101
            }
        )
        assert response.status_code == 422
    
    # ============= Health Check Test =============
    
    def test_health_check(self):
        """Test RAG service health check endpoint"""
        response = self.client.get("/api/v1/rag/health", headers=self.auth_headers)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "rag_v2"
        assert "timestamp" in data
    
    # ============= Service Manager Tests =============
    
    @pytest.mark.asyncio
    @patch('tldw_Server_API.app.api.v1.endpoints.rag_v2.RAGService')
    async def test_rag_service_caching(self, mock_rag_service):
        """Test that RAG services are cached per user"""
        from tldw_Server_API.app.api.v1.endpoints.rag_v2 import rag_service_manager
        
        # Clear cache
        rag_service_manager._cache.clear()
        
        # Mock service
        mock_service = AsyncMock()
        mock_service.initialize = AsyncMock()
        mock_service.cleanup = Mock()
        mock_rag_service.return_value = mock_service
        
        # Create service for user 1
        service1 = await rag_service_manager.get_or_create(
            user_id=1,
            media_db_path=Path("/test/media1.db"),
            chacha_db_path=Path("/test/chacha1.db")
        )
        
        # Get same service for user 1 - should be cached
        service1_cached = await rag_service_manager.get_or_create(
            user_id=1,
            media_db_path=Path("/test/media1.db"),
            chacha_db_path=Path("/test/chacha1.db")
        )
        
        assert service1 is service1_cached
        assert mock_rag_service.call_count == 1  # Only created once
        
        # Create service for user 2 - should be different
        service2 = await rag_service_manager.get_or_create(
            user_id=2,
            media_db_path=Path("/test/media2.db"),
            chacha_db_path=Path("/test/chacha2.db")
        )
        
        assert service2 is not service1
        assert mock_rag_service.call_count == 2  # Created twice total
    
    @pytest.mark.asyncio
    @patch('tldw_Server_API.app.api.v1.endpoints.rag_v2.RAGService')
    async def test_rag_service_cleanup(self, mock_rag_service):
        """Test cleanup of expired RAG services"""
        from tldw_Server_API.app.api.v1.endpoints.rag_v2 import rag_service_manager
        import time
        
        # Clear cache
        rag_service_manager._cache.clear()
        
        # Mock service
        mock_service = AsyncMock()
        mock_service.initialize = AsyncMock()
        mock_service.cleanup = Mock()
        mock_rag_service.return_value = mock_service
        
        # Create service with short TTL
        rag_service_manager._ttl = 0.1  # 100ms TTL for testing
        
        service = await rag_service_manager.get_or_create(
            user_id=1,
            media_db_path=Path("/test/media.db"),
            chacha_db_path=Path("/test/chacha.db")
        )
        
        # Wait for expiration
        await asyncio.sleep(0.2)
        
        # Cleanup expired services
        await rag_service_manager.cleanup_expired()
        
        # Verify service was cleaned up
        mock_service.cleanup.assert_called_once()
        assert 1 not in rag_service_manager._cache
        
        # Reset TTL
        rag_service_manager._ttl = 3600