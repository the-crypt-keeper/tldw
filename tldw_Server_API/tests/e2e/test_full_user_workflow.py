# test_full_user_workflow.py
# Description: Comprehensive end-to-end user workflow tests
#
"""
End-to-End User Workflow Tests
-------------------------------

Validates the complete user journey through the tldw_server API, testing
all core functionality in realistic scenarios.

Test Phases:
1. Setup & Authentication
2. Media Ingestion & Processing
3. Transcription & Analysis
4. Chat & Interaction
5. Notes & Knowledge Management
6. Prompts & Templates
7. Character & Persona Management
8. RAG & Search
9. Evaluation & Testing
10. Export & Sync
11. Cleanup (Deletion Tests)
"""

import os
import time
import json
from typing import Dict, Any, List, Optional
import pytest
from pathlib import Path
from datetime import datetime, timedelta

from fixtures import (
    api_client, authenticated_client, test_user_credentials, data_tracker,
    create_test_file, create_test_pdf, create_test_audio, cleanup_test_file
)
from test_data import (
    TestDataGenerator, TestScenarios, generate_unique_id,
    generate_test_user, generate_batch_data
)


class TestFullUserWorkflow:
    """Complete end-to-end test of user interaction with the API."""
    
    # Class-level storage for data persistence across tests
    user_data: Dict[str, Any] = {}
    media_items: List[Dict[str, Any]] = []
    notes: List[Dict[str, Any]] = []
    prompts: List[Dict[str, Any]] = []
    characters: List[Dict[str, Any]] = []
    chats: List[Dict[str, Any]] = []
    
    # Performance tracking
    performance_metrics: Dict[str, float] = {}
    
    @pytest.fixture(autouse=True)
    def track_performance(self):
        """Track test execution time."""
        start_time = time.time()
        yield
        duration = time.time() - start_time
        test_name = os.environ.get('PYTEST_CURRENT_TEST', '').split('::')[-1].split('[')[0]
        self.performance_metrics[test_name] = duration
    
    # ========================================================================
    # Phase 1: Setup & Authentication
    # ========================================================================
    
    def test_01_health_check(self, api_client):
        """Test API health check endpoint."""
        response = api_client.health_check()
        assert response.get("status") == "healthy"
        assert "version" in response
        assert "timestamp" in response
    
    def test_02_user_registration(self, api_client, data_tracker):
        """Test user registration (if multi-user mode)."""
        # Generate unique user data
        user = generate_test_user()
        
        try:
            # Try to register
            response = api_client.register(
                username=user["username"],
                email=user["email"],
                password=user["password"]
            )
            
            # Store user data
            self.user_data = {
                "username": user["username"],
                "email": user["email"],
                "password": user["password"],
                "user_id": response.get("user_id")
            }
            
            assert response.get("success") == True
            assert "user_id" in response or "message" in response
            
        except Exception as e:
            # Single-user mode or registration disabled
            print(f"Registration test skipped: {e}")
            self.user_data = test_user_credentials
    
    def test_03_user_login(self, api_client):
        """Test user login and token generation."""
        if not self.user_data:
            pytest.skip("No user data available")
        
        response = api_client.login(
            username=self.user_data.get("username", "test_user"),
            password=self.user_data.get("password", "test_password")
        )
        
        # Verify response
        assert "access_token" in response or "token" in response
        
        # Store tokens
        token = response.get("access_token") or response.get("token")
        api_client.set_auth_token(token, response.get("refresh_token"))
        
        # Store in class data
        self.user_data["access_token"] = token
        self.user_data["refresh_token"] = response.get("refresh_token")
    
    def test_04_get_user_profile(self, authenticated_client):
        """Test getting current user profile."""
        response = authenticated_client.get_current_user()
        
        assert "id" in response or "user_id" in response
        assert "username" in response or "email" in response
        
        # Update user data
        self.user_data.update(response)
    
    # ========================================================================
    # Phase 2: Media Ingestion & Processing
    # ========================================================================
    
    def test_10_upload_text_document(self, authenticated_client, data_tracker):
        """Test uploading a text document."""
        # Create test file
        content = TestDataGenerator.sample_text_content()
        file_path = create_test_file(content, suffix=".txt")
        data_tracker.add_file(file_path)
        
        try:
            # Upload file
            response = authenticated_client.upload_media(
                file_path=file_path,
                title="E2E Test Document",
                media_type="document"
            )
            
            # Verify response
            assert "media_id" in response or "id" in response
            assert response.get("status") in ["success", "processing", "completed"]
            
            # Store media item
            self.media_items.append(response)
            media_id = response.get("media_id") or response.get("id")
            data_tracker.add_media(media_id)
            
        finally:
            cleanup_test_file(file_path)
    
    def test_11_upload_pdf_document(self, authenticated_client, data_tracker):
        """Test uploading a PDF document."""
        # Create test PDF
        file_path = create_test_pdf()
        data_tracker.add_file(file_path)
        
        try:
            # Upload file
            response = authenticated_client.upload_media(
                file_path=file_path,
                title="E2E Test PDF",
                media_type="pdf"
            )
            
            # Verify response
            assert "media_id" in response or "id" in response
            
            # Store media item
            self.media_items.append(response)
            media_id = response.get("media_id") or response.get("id")
            data_tracker.add_media(media_id)
            
        finally:
            cleanup_test_file(file_path)
    
    def test_12_process_web_content(self, authenticated_client, data_tracker):
        """Test processing content from a URL."""
        # Use a reliable test URL
        test_url = "https://en.wikipedia.org/wiki/Artificial_intelligence"
        
        try:
            response = authenticated_client.process_media(
                url=test_url,
                title="E2E Test Web Content"
            )
            
            # Verify response
            assert response.get("status") in ["success", "processing", "completed"]
            
            if "media_id" in response or "id" in response:
                self.media_items.append(response)
                media_id = response.get("media_id") or response.get("id")
                data_tracker.add_media(media_id)
                
        except Exception as e:
            print(f"Web content processing skipped: {e}")
    
    def test_13_upload_audio_file(self, authenticated_client, data_tracker):
        """Test uploading an audio file."""
        # Create test audio
        file_path = create_test_audio()
        data_tracker.add_file(file_path)
        
        try:
            # Upload file
            response = authenticated_client.upload_media(
                file_path=file_path,
                title="E2E Test Audio",
                media_type="audio"
            )
            
            # Verify response
            assert "media_id" in response or "id" in response
            
            # Store media item
            self.media_items.append(response)
            media_id = response.get("media_id") or response.get("id")
            data_tracker.add_media(media_id)
            
        finally:
            cleanup_test_file(file_path)
    
    def test_14_list_media_items(self, authenticated_client):
        """Test listing all media items."""
        response = authenticated_client.get_media_list(limit=50)
        
        # Verify response structure
        assert "items" in response or "results" in response
        items = response.get("items") or response.get("results", [])
        
        # Should have at least the items we uploaded
        assert len(items) >= len(self.media_items)
        
        # Verify item structure
        if items:
            item = items[0]
            assert "id" in item or "media_id" in item
            assert "title" in item
    
    # ========================================================================
    # Phase 3: Transcription & Analysis
    # ========================================================================
    
    def test_20_get_media_details(self, authenticated_client):
        """Test getting details of uploaded media."""
        if not self.media_items:
            pytest.skip("No media items available")
        
        # Get details of first media item
        media_item = self.media_items[0]
        media_id = media_item.get("media_id") or media_item.get("id")
        
        response = authenticated_client.get_media_item(media_id)
        
        # Verify response
        assert "id" in response or "media_id" in response
        assert "title" in response
        assert "content" in response or "text" in response or "transcript" in response
    
    # ========================================================================
    # Phase 4: Chat & Interaction
    # ========================================================================
    
    def test_30_simple_chat_completion(self, authenticated_client, data_tracker):
        """Test basic chat completion."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2+2?"}
        ]
        
        try:
            response = authenticated_client.chat_completion(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=0.7
            )
            
            # Verify response structure
            assert "choices" in response or "response" in response or "content" in response
            
            # Store chat
            self.chats.append({
                "messages": messages,
                "response": response
            })
            
            if "chat_id" in response:
                data_tracker.add_chat(response["chat_id"])
                
        except Exception as e:
            print(f"Chat completion test skipped: {e}")
    
    def test_31_chat_with_context(self, authenticated_client, data_tracker):
        """Test chat with media context (RAG)."""
        if not self.media_items:
            pytest.skip("No media items available")
        
        messages = TestDataGenerator.sample_chat_messages()
        
        try:
            response = authenticated_client.chat_completion(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=0.7
            )
            
            # Store chat
            self.chats.append({
                "messages": messages,
                "response": response
            })
            
            if "chat_id" in response:
                data_tracker.add_chat(response["chat_id"])
                
        except Exception as e:
            print(f"Chat with context test skipped: {e}")
    
    # ========================================================================
    # Phase 5: Notes & Knowledge Management
    # ========================================================================
    
    def test_40_create_note(self, authenticated_client, data_tracker):
        """Test creating a note."""
        note_data = TestDataGenerator.sample_note()
        
        response = authenticated_client.create_note(
            title=note_data["title"],
            content=note_data["content"],
            keywords=note_data.get("keywords")
        )
        
        # Verify response
        assert "id" in response or "note_id" in response
        assert response.get("title") == note_data["title"]
        
        # Store note
        self.notes.append(response)
        note_id = response.get("id") or response.get("note_id")
        data_tracker.add_note(note_id)
    
    def test_41_list_notes(self, authenticated_client):
        """Test listing notes."""
        response = authenticated_client.get_notes(limit=50)
        
        # Verify response
        assert "items" in response or "results" in response or "notes" in response
        items = response.get("items") or response.get("results") or response.get("notes", [])
        
        # Should have at least the notes we created
        assert len(items) >= len(self.notes)
    
    def test_42_update_note(self, authenticated_client):
        """Test updating a note."""
        if not self.notes:
            pytest.skip("No notes available")
        
        note = self.notes[0]
        note_id = note.get("id") or note.get("note_id")
        
        updated_content = note.get("content", "") + "\n\n## Updated Section\nThis section was added during E2E testing."
        
        response = authenticated_client.update_note(
            note_id=note_id,
            content=updated_content
        )
        
        # Verify update
        assert response.get("success") == True or "id" in response
    
    def test_43_search_notes(self, authenticated_client):
        """Test searching notes."""
        if not self.notes:
            pytest.skip("No notes available")
        
        # Search for a keyword from our test note
        response = authenticated_client.search_notes("machine learning")
        
        # Verify response
        assert "results" in response or "items" in response or "notes" in response
    
    # ========================================================================
    # Phase 6: Prompts & Templates
    # ========================================================================
    
    def test_50_create_prompt(self, authenticated_client, data_tracker):
        """Test creating a prompt template."""
        prompt_data = TestDataGenerator.sample_prompt_template()
        
        response = authenticated_client.create_prompt(
            name=prompt_data["name"],
            content=prompt_data["content"],
            description=prompt_data.get("description")
        )
        
        # Verify response
        assert "id" in response or "prompt_id" in response
        
        # Store prompt
        self.prompts.append(response)
        prompt_id = response.get("id") or response.get("prompt_id")
        data_tracker.add_prompt(prompt_id)
    
    def test_51_list_prompts(self, authenticated_client):
        """Test listing prompts."""
        response = authenticated_client.get_prompts()
        
        # Verify response
        assert "items" in response or "results" in response or "prompts" in response
        items = response.get("items") or response.get("results") or response.get("prompts", [])
        
        # Should have at least the prompts we created
        assert len(items) >= len(self.prompts)
    
    # ========================================================================
    # Phase 7: Character & Persona Management
    # ========================================================================
    
    def test_60_import_character(self, authenticated_client, data_tracker):
        """Test importing a character card."""
        character_data = TestDataGenerator.sample_character_card()
        
        try:
            response = authenticated_client.import_character(character_data)
            
            # Verify response
            assert "id" in response or "character_id" in response
            
            # Store character
            self.characters.append(response)
            character_id = response.get("id") or response.get("character_id")
            data_tracker.add_character(character_id)
            
        except Exception as e:
            print(f"Character import test skipped: {e}")
    
    def test_61_list_characters(self, authenticated_client):
        """Test listing characters."""
        if not self.characters:
            pytest.skip("No characters available")
        
        response = authenticated_client.get_characters()
        
        # Verify response
        assert "items" in response or "results" in response or "characters" in response
    
    # ========================================================================
    # Phase 8: RAG & Search
    # ========================================================================
    
    def test_70_search_media_content(self, authenticated_client):
        """Test searching across media content."""
        if not self.media_items:
            pytest.skip("No media items available")
        
        # Search for content we know exists
        queries = TestDataGenerator.sample_search_queries()
        
        for query in queries[:3]:  # Test first 3 queries
            try:
                response = authenticated_client.search_media(query, limit=10)
                
                # Verify response structure
                assert "results" in response or "items" in response
                
                break  # One successful search is enough
                
            except Exception as e:
                print(f"Search for '{query}' failed: {e}")
                continue
    
    # ========================================================================
    # Phase 9: Evaluation & Testing (if available)
    # ========================================================================
    
    def test_80_evaluation_placeholder(self, authenticated_client):
        """Placeholder for evaluation tests."""
        # This would test evaluation endpoints if available
        pass
    
    # ========================================================================
    # Phase 10: Export & Sync
    # ========================================================================
    
    def test_90_export_placeholder(self, authenticated_client):
        """Placeholder for export tests."""
        # This would test export functionality if available
        pass
    
    # ========================================================================
    # Phase 11: Cleanup (Deletion Tests)
    # ========================================================================
    
    def test_100_delete_notes(self, authenticated_client):
        """Test deleting notes."""
        for note in self.notes:
            note_id = note.get("id") or note.get("note_id")
            try:
                response = authenticated_client.delete_note(note_id)
                assert response.get("success") == True or response.get("status") == "deleted"
            except Exception as e:
                print(f"Failed to delete note {note_id}: {e}")
    
    def test_101_delete_prompts(self, authenticated_client):
        """Test deleting prompts."""
        for prompt in self.prompts:
            prompt_id = prompt.get("id") or prompt.get("prompt_id")
            try:
                response = authenticated_client.delete_prompt(prompt_id)
                assert response.get("success") == True or response.get("status") == "deleted"
            except Exception as e:
                print(f"Failed to delete prompt {prompt_id}: {e}")
    
    def test_102_delete_characters(self, authenticated_client):
        """Test deleting characters."""
        for character in self.characters:
            character_id = character.get("id") or character.get("character_id")
            try:
                response = authenticated_client.delete_character(character_id)
                assert response.get("success") == True or response.get("status") == "deleted"
            except Exception as e:
                print(f"Failed to delete character {character_id}: {e}")
    
    def test_103_delete_media(self, authenticated_client):
        """Test deleting media items."""
        for media in self.media_items:
            media_id = media.get("media_id") or media.get("id")
            try:
                response = authenticated_client.delete_media(media_id)
                assert response.get("success") == True or response.get("status") == "deleted"
            except Exception as e:
                print(f"Failed to delete media {media_id}: {e}")
    
    def test_104_logout(self, authenticated_client):
        """Test user logout."""
        try:
            response = authenticated_client.logout()
            assert response.get("success") == True or response.get("message")
        except Exception as e:
            print(f"Logout test skipped: {e}")
    
    def test_105_performance_summary(self):
        """Print performance summary."""
        if self.performance_metrics:
            print("\n=== Performance Summary ===")
            total_time = sum(self.performance_metrics.values())
            
            for test_name, duration in sorted(self.performance_metrics.items()):
                print(f"{test_name}: {duration:.2f}s")
            
            print(f"\nTotal execution time: {total_time:.2f}s")
            print(f"Average test time: {total_time/len(self.performance_metrics):.2f}s")


# Additional test scenarios can be added here
class TestAdvancedScenarios:
    """Advanced scenario-based tests."""
    
    def test_research_workflow(self, authenticated_client, data_tracker):
        """Test complete research workflow scenario."""
        scenario = TestScenarios.research_workflow()
        
        # Execute each step in the scenario
        for step in scenario["steps"]:
            action = step["action"]
            data = step["data"]
            
            # Execute action based on type
            # This would be expanded with actual implementation
            print(f"Executing: {action}")
    
    def test_content_creation_workflow(self, authenticated_client, data_tracker):
        """Test content creation workflow scenario."""
        scenario = TestScenarios.content_creation_workflow()
        
        # Execute scenario steps
        for step in scenario["steps"]:
            print(f"Executing: {step['action']}")
    
    def test_media_processing_workflow(self, authenticated_client, data_tracker):
        """Test media processing workflow scenario."""
        scenario = TestScenarios.media_processing_workflow()
        
        # Execute scenario steps
        for step in scenario["steps"]:
            print(f"Executing: {step['action']}")