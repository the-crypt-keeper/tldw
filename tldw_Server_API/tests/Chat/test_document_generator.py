# test_document_generator.py
# Description: Unit tests for the DocumentGeneratorService
#
"""
Document Generator Service Tests
---------------------------------

Comprehensive unit tests for document generation functionality including
all document types, async job management, and prompt customization.
"""

import pytest
import json
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock, call
from datetime import datetime
from uuid import uuid4

from tldw_Server_API.app.core.Chat.document_generator import DocumentGeneratorService, DocumentType
from tldw_Server_API.app.core.DB_Management.ChaChaNotes_DB import CharactersRAGDB


@pytest.fixture
def mock_db():
    """Create a mock database instance."""
    mock = MagicMock()
    mock.execute_query = MagicMock()
    mock.get_conversation_by_id = MagicMock()
    mock.get_messages_for_conversation = MagicMock()
    return mock


@pytest.fixture
def service(mock_db):
    """Create a DocumentGeneratorService instance with mocked database."""
    return DocumentGeneratorService(mock_db, user_id="test_user")


@pytest.fixture
def sample_conversation():
    """Create sample conversation data."""
    return {
        "id": "conv123",
        "title": "Test Conversation",
        "created_at": "2024-01-01T00:00:00Z",
        "messages": [
            {"sender": "user", "content": "Hello, how are you?", "timestamp": "2024-01-01T00:00:00Z"},
            {"sender": "assistant", "content": "I'm doing well, thank you!", "timestamp": "2024-01-01T00:01:00Z"},
            {"sender": "user", "content": "Can you explain quantum computing?", "timestamp": "2024-01-01T00:02:00Z"},
            {"sender": "assistant", "content": "Quantum computing uses quantum bits...", "timestamp": "2024-01-01T00:03:00Z"}
        ]
    }


class TestDocumentGeneratorService:
    """Test suite for DocumentGeneratorService."""
    
    def test_init_creates_tables(self, mock_db):
        """Test that initialization creates necessary tables."""
        service = DocumentGeneratorService(mock_db, "test_user")
        
        # Should create tables for documents and jobs
        assert mock_db.execute_query.call_count >= 2
        
        # Check table creation
        calls = mock_db.execute_query.call_args_list
        sql_statements = [call[0][0] for call in calls]
        assert any("CREATE TABLE IF NOT EXISTS generated_documents" in sql for sql in sql_statements)
        assert any("CREATE TABLE IF NOT EXISTS generation_jobs" in sql for sql in sql_statements)
    
    @pytest.mark.asyncio
    async def test_generate_timeline(self, service, mock_db, sample_conversation):
        """Test timeline document generation."""
        mock_db.get_conversation_by_id.return_value = sample_conversation
        mock_db.get_messages_for_conversation.return_value = sample_conversation["messages"]
        mock_db.execute_query.return_value = [{"id": "doc123"}]
        
        # Mock LLM call
        with patch.object(service, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Timeline: Event 1, Event 2, Event 3"
            
            result = await service.generate_document(
                conversation_id="conv123",
                document_type=DocumentType.TIMELINE,
                custom_prompt=None
            )
        
        assert result["success"] == True
        assert result["document_id"] == "doc123"
        assert result["document_type"] == DocumentType.TIMELINE
        mock_llm.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_generate_study_guide(self, service, mock_db, sample_conversation):
        """Test study guide document generation."""
        mock_db.get_conversation_by_id.return_value = sample_conversation
        mock_db.get_messages_for_conversation.return_value = sample_conversation["messages"]
        mock_db.execute_query.return_value = [{"id": "doc124"}]
        
        with patch.object(service, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Study Guide: Key concepts and review questions"
            
            result = await service.generate_document(
                conversation_id="conv123",
                document_type=DocumentType.STUDY_GUIDE
            )
        
        assert result["success"] == True
        assert result["document_type"] == DocumentType.STUDY_GUIDE
        
        # Check that study guide prompt was used
        llm_call = mock_llm.call_args[0][0]
        assert "study guide" in llm_call.lower()
    
    @pytest.mark.asyncio
    async def test_generate_briefing(self, service, mock_db, sample_conversation):
        """Test executive briefing document generation."""
        mock_db.get_conversation_by_id.return_value = sample_conversation
        mock_db.get_messages_for_conversation.return_value = sample_conversation["messages"]
        mock_db.execute_query.return_value = [{"id": "doc125"}]
        
        with patch.object(service, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Executive Summary: Key points and recommendations"
            
            result = await service.generate_document(
                conversation_id="conv123",
                document_type=DocumentType.BRIEFING
            )
        
        assert result["success"] == True
        assert "Executive Summary" in result["content"]
    
    @pytest.mark.asyncio
    async def test_generate_summary(self, service, mock_db, sample_conversation):
        """Test summary document generation."""
        mock_db.get_conversation_by_id.return_value = sample_conversation
        mock_db.get_messages_for_conversation.return_value = sample_conversation["messages"]
        mock_db.execute_query.return_value = [{"id": "doc126"}]
        
        with patch.object(service, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Summary: Main discussion points"
            
            result = await service.generate_document(
                conversation_id="conv123",
                document_type=DocumentType.SUMMARY
            )
        
        assert result["success"] == True
        assert result["document_type"] == DocumentType.SUMMARY
    
    @pytest.mark.asyncio
    async def test_generate_qa_pairs(self, service, mock_db, sample_conversation):
        """Test Q&A pairs document generation."""
        mock_db.get_conversation_by_id.return_value = sample_conversation
        mock_db.get_messages_for_conversation.return_value = sample_conversation["messages"]
        mock_db.execute_query.return_value = [{"id": "doc127"}]
        
        with patch.object(service, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Q1: Question?\nA1: Answer."
            
            result = await service.generate_document(
                conversation_id="conv123",
                document_type=DocumentType.QA_PAIRS
            )
        
        assert result["success"] == True
        assert "Q1:" in result["content"]
    
    @pytest.mark.asyncio
    async def test_generate_meeting_notes(self, service, mock_db, sample_conversation):
        """Test meeting notes document generation."""
        mock_db.get_conversation_by_id.return_value = sample_conversation
        mock_db.get_messages_for_conversation.return_value = sample_conversation["messages"]
        mock_db.execute_query.return_value = [{"id": "doc128"}]
        
        with patch.object(service, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Meeting Notes: Attendees, Agenda, Action Items"
            
            result = await service.generate_document(
                conversation_id="conv123",
                document_type=DocumentType.MEETING_NOTES
            )
        
        assert result["success"] == True
        assert "Meeting Notes" in result["content"]
    
    @pytest.mark.asyncio
    async def test_custom_prompt(self, service, mock_db, sample_conversation):
        """Test using custom prompt for generation."""
        mock_db.get_conversation_by_id.return_value = sample_conversation
        mock_db.get_messages_for_conversation.return_value = sample_conversation["messages"]
        mock_db.execute_query.return_value = [{"id": "doc129"}]
        
        custom_prompt = "Extract only the technical terms mentioned"
        
        with patch.object(service, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Technical terms: quantum computing, quantum bits"
            
            result = await service.generate_document(
                conversation_id="conv123",
                document_type=DocumentType.SUMMARY,
                custom_prompt=custom_prompt
            )
        
        # Custom prompt should be used
        llm_call = mock_llm.call_args[0][0]
        assert custom_prompt in llm_call
    
    def test_async_job_creation(self, service, mock_db):
        """Test creating an async generation job."""
        job_id = str(uuid4())
        
        with patch('uuid.uuid4', return_value=job_id):
            mock_db.execute_query.return_value = None
            
            result = service.create_generation_job(
                conversation_id="conv123",
                document_type=DocumentType.TIMELINE
            )
        
        assert result["job_id"] == job_id
        assert result["status"] == "pending"
        
        # Check job was saved to database
        mock_db.execute_query.assert_called()
        call_args = mock_db.execute_query.call_args[0]
        assert "INSERT INTO generation_jobs" in call_args[0]
    
    def test_get_job_status(self, service, mock_db):
        """Test retrieving job status."""
        mock_db.execute_query.return_value = [{
            "job_id": "job123",
            "status": "completed",
            "document_id": "doc123",
            "created_at": "2024-01-01T00:00:00Z",
            "completed_at": "2024-01-01T00:01:00Z",
            "error_message": None
        }]
        
        result = service.get_job_status("job123")
        
        assert result["job_id"] == "job123"
        assert result["status"] == "completed"
        assert result["document_id"] == "doc123"
    
    def test_cancel_job(self, service, mock_db):
        """Test cancelling a generation job."""
        mock_db.execute_query.return_value = None
        
        result = service.cancel_job("job123")
        
        assert result == True
        
        # Check job was updated
        call_args = mock_db.execute_query.call_args[0]
        assert "UPDATE generation_jobs SET status = ?" in call_args[0]
        assert "cancelled" in call_args[1]
    
    def test_get_document(self, service, mock_db):
        """Test retrieving a generated document."""
        mock_db.execute_query.return_value = [{
            "id": "doc123",
            "conversation_id": "conv123",
            "document_type": "timeline",
            "title": "Conversation Timeline",
            "content": "Timeline content",
            "created_at": "2024-01-01T00:00:00Z",
            "metadata": json.dumps({"word_count": 100})
        }]
        
        result = service.get_document("doc123")
        
        assert result["id"] == "doc123"
        assert result["document_type"] == "timeline"
        assert result["metadata"]["word_count"] == 100
    
    def test_list_documents(self, service, mock_db):
        """Test listing generated documents."""
        mock_db.execute_query.return_value = [
            {"id": "doc1", "title": "Timeline", "document_type": "timeline"},
            {"id": "doc2", "title": "Summary", "document_type": "summary"}
        ]
        
        results = service.list_documents(conversation_id="conv123")
        
        assert len(results) == 2
        assert results[0]["document_type"] == "timeline"
        assert results[1]["document_type"] == "summary"
    
    def test_delete_document(self, service, mock_db):
        """Test deleting a generated document."""
        mock_db.execute_query.return_value = None
        
        result = service.delete_document("doc123")
        
        assert result == True
        
        # Check document was deleted
        call_args = mock_db.execute_query.call_args[0]
        assert "DELETE FROM generated_documents WHERE id = ?" in call_args[0]
        assert "doc123" in call_args[1]
    
    def test_save_custom_prompt_config(self, service, mock_db):
        """Test saving custom prompt configuration."""
        mock_db.execute_query.return_value = None
        
        config = {
            DocumentType.TIMELINE: "Custom timeline prompt",
            DocumentType.SUMMARY: "Custom summary prompt"
        }
        
        result = service.save_prompt_config(config)
        
        assert result == True
        
        # Check config was saved
        call_args = mock_db.execute_query.call_args[0]
        assert "INSERT OR REPLACE INTO user_prompt_configs" in call_args[0]
    
    def test_get_prompt_config(self, service, mock_db):
        """Test retrieving custom prompt configuration."""
        mock_db.execute_query.return_value = [{
            "document_type": "timeline",
            "custom_prompt": "Custom timeline prompt"
        }]
        
        result = service.get_prompt_config(DocumentType.TIMELINE)
        
        assert result == "Custom timeline prompt"
    
    @pytest.mark.asyncio
    async def test_bulk_generation(self, service, mock_db, sample_conversation):
        """Test generating multiple document types at once."""
        mock_db.get_conversation_by_id.return_value = sample_conversation
        mock_db.get_messages_for_conversation.return_value = sample_conversation["messages"]
        mock_db.execute_query.return_value = [{"id": "doc_bulk"}]
        
        with patch.object(service, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = "Generated content"
            
            results = await service.bulk_generate(
                conversation_id="conv123",
                document_types=[DocumentType.TIMELINE, DocumentType.SUMMARY]
            )
        
        assert len(results) == 2
        assert all(r["success"] for r in results)
        assert mock_llm.call_count == 2
    
    def test_get_statistics(self, service, mock_db):
        """Test getting generation statistics."""
        mock_db.execute_query.side_effect = [
            [{"total_documents": 100}],
            [{"document_type": "timeline", "count": 40}],
            [{"total_jobs": 150, "completed_jobs": 140, "failed_jobs": 5}]
        ]
        
        stats = service.get_statistics()
        
        assert stats["total_documents"] == 100
        assert stats["documents_by_type"]["timeline"] == 40
        assert stats["total_jobs"] == 150
        assert stats["success_rate"] == 140 / 150
    
    @pytest.mark.asyncio
    async def test_error_handling(self, service, mock_db, sample_conversation):
        """Test error handling during generation."""
        mock_db.get_conversation_by_id.return_value = sample_conversation
        mock_db.get_messages_for_conversation.return_value = sample_conversation["messages"]
        
        with patch.object(service, '_call_llm', new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("LLM API error")
            
            result = await service.generate_document(
                conversation_id="conv123",
                document_type=DocumentType.TIMELINE
            )
        
        assert result["success"] == False
        assert "error" in result
        assert "LLM API error" in result["error"]
    
    @pytest.mark.asyncio
    async def test_conversation_not_found(self, service, mock_db):
        """Test handling when conversation doesn't exist."""
        mock_db.get_conversation_by_id.return_value = None
        
        result = await service.generate_document(
            conversation_id="nonexistent",
            document_type=DocumentType.TIMELINE
        )
        
        assert result["success"] == False
        assert "not found" in result["error"].lower()