# fixtures.py
# Description: Shared fixtures and utilities for end-to-end tests
#
"""
End-to-End Test Fixtures and Utilities
---------------------------------------

Provides common fixtures, utilities, and helper functions for the comprehensive
end-to-end test suite.
"""

import os
import json
import base64
import tempfile
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta
import pytest
import httpx
from pathlib import Path

# Configuration
BASE_URL = os.getenv("E2E_TEST_BASE_URL", "http://localhost:8000")
API_PREFIX = "/api/v1"
TEST_TIMEOUT = 120  # seconds for each request (increased for video transcription)


class APIClient:
    """Wrapper for API interactions with authentication support."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=TEST_TIMEOUT)
        self.token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.user_id: Optional[int] = None
        
    def set_auth_token(self, token: str, refresh_token: Optional[str] = None):
        """Set authentication tokens."""
        self.token = token
        self.refresh_token = refresh_token
        # In single-user mode, use X-API-KEY header
        # Also add Token header for some endpoints that expect it (case-sensitive)
        # Add Bearer authorization for OpenAI-compatible endpoints
        self.client.headers.update({
            "X-API-KEY": token,
            "Token": token,  # Some endpoints expect this (capital T)
            "Authorization": f"Bearer {token}"  # For OpenAI-compatible endpoints
        })
    
    def clear_auth(self):
        """Clear authentication."""
        self.token = None
        self.refresh_token = None
        if "Authorization" in self.client.headers:
            del self.client.headers["Authorization"]
        if "X-API-KEY" in self.client.headers:
            del self.client.headers["X-API-KEY"]
    
    # Authentication endpoints
    def register(self, username: str, email: str, password: str) -> Dict[str, Any]:
        """Register a new user."""
        response = self.client.post(
            f"{API_PREFIX}/auth/register",
            json={
                "username": username,
                "email": email,
                "password": password
            }
        )
        response.raise_for_status()
        return response.json()
    
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """Login and obtain tokens."""
        response = self.client.post(
            f"{API_PREFIX}/auth/login",
            json={
                "username": username,
                "password": password
            }
        )
        response.raise_for_status()
        data = response.json()
        
        # Automatically set tokens
        if "access_token" in data:
            self.set_auth_token(data["access_token"], data.get("refresh_token"))
            
        return data
    
    def logout(self) -> Dict[str, Any]:
        """Logout current user."""
        response = self.client.post(f"{API_PREFIX}/auth/logout")
        response.raise_for_status()
        self.clear_auth()
        return response.json()
    
    def get_current_user(self) -> Dict[str, Any]:
        """Get current user information."""
        try:
            response = self.client.get(f"{API_PREFIX}/auth/me")
            response.raise_for_status()
            data = response.json()
            if "id" in data:
                self.user_id = data["id"]
            return data
        except httpx.HTTPStatusError:
            # In single-user mode, this might not be available
            return {"username": "single_user", "id": 1}
    
    # Media endpoints
    def upload_media(self, file_path: str, title: str, media_type: str = "document") -> Dict[str, Any]:
        """Upload a media file."""
        with open(file_path, "rb") as f:
            # The endpoint expects 'files' (plural) not 'file'
            files = {"files": (os.path.basename(file_path), f, "application/octet-stream")}
            data = {
                "title": title,
                "media_type": media_type,
                "overwrite_existing": "false",
                "keep_original_file": "false"
            }
            response = self.client.post(
                f"{API_PREFIX}/media/add",
                files=files,
                data=data
            )
        if response.status_code != 200:
            print(f"Error response: {response.text}")
        response.raise_for_status()
        return response.json()
    
    def process_media(self, url: Optional[str] = None, file_path: Optional[str] = None,
                     title: Optional[str] = None, custom_prompt: Optional[str] = None) -> Dict[str, Any]:
        """Process media from URL or file."""
        data = {}
        files = None
        
        if url:
            data["url"] = url
        if title:
            data["title"] = title
        if custom_prompt:
            data["custom_prompt"] = custom_prompt
            
        if file_path:
            with open(file_path, "rb") as f:
                files = {"file": (os.path.basename(file_path), f, "application/octet-stream")}
                
        response = self.client.post(
            f"{API_PREFIX}/media/process",
            data=data if not files else None,
            json=data if files else None,
            files=files
        )
        response.raise_for_status()
        return response.json()
    
    def get_media_list(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Get list of media items."""
        response = self.client.get(
            f"{API_PREFIX}/media/",
            params={"limit": limit, "offset": offset}
        )
        response.raise_for_status()
        return response.json()
    
    def get_media_item(self, media_id: int) -> Dict[str, Any]:
        """Get specific media item details."""
        response = self.client.get(f"{API_PREFIX}/media/{media_id}")
        response.raise_for_status()
        return response.json()
    
    def delete_media(self, media_id: int) -> Dict[str, Any]:
        """Delete a media item."""
        response = self.client.delete(f"{API_PREFIX}/media/{media_id}")
        response.raise_for_status()
        return response.json()
    
    # Chat endpoints
    def chat_completion(self, messages: List[Dict[str, str]], 
                        model: str = "gpt-3.5-turbo",
                        temperature: float = 0.7,
                        stream: bool = False) -> Dict[str, Any]:
        """Send chat completion request."""
        response = self.client.post(
            f"{API_PREFIX}/chat/completions",
            json={
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "stream": stream
            }
        )
        response.raise_for_status()
        return response.json()
    
    # Notes endpoints
    def create_note(self, title: str, content: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a new note."""
        data = {
            "title": title,
            "content": content
        }
        if keywords:
            data["keywords"] = keywords
            
        response = self.client.post(f"{API_PREFIX}/notes/", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_notes(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Get list of notes."""
        response = self.client.get(
            f"{API_PREFIX}/notes/",
            params={"limit": limit, "offset": offset}
        )
        response.raise_for_status()
        return response.json()
    
    def update_note(self, note_id: str, title: Optional[str] = None, 
                   content: Optional[str] = None, version: int = 1) -> Dict[str, Any]:
        """Update an existing note."""
        data = {}
        if title:
            data["title"] = title
        if content:
            data["content"] = content
            
        # Add expected version header for optimistic locking
        headers = {"expected-version": str(version)}
        response = self.client.put(f"{API_PREFIX}/notes/{note_id}", json=data, headers=headers)
        response.raise_for_status()
        return response.json()
    
    def delete_note(self, note_id: int) -> Dict[str, Any]:
        """Delete a note."""
        response = self.client.delete(f"{API_PREFIX}/notes/{note_id}")
        response.raise_for_status()
        return response.json()
    
    def search_notes(self, query: str) -> Dict[str, Any]:
        """Search notes."""
        response = self.client.get(
            f"{API_PREFIX}/notes/search/",
            params={"query": query}
        )
        response.raise_for_status()
        return response.json()
    
    # Prompts endpoints
    def create_prompt(self, name: str, content: str, description: Optional[str] = None) -> Dict[str, Any]:
        """Create a new prompt."""
        data = {
            "name": name,
            "content": content
        }
        if description:
            data["description"] = description
            
        response = self.client.post(f"{API_PREFIX}/prompts/", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_prompts(self) -> Dict[str, Any]:
        """Get list of prompts."""
        response = self.client.get(f"{API_PREFIX}/prompts/")
        response.raise_for_status()
        return response.json()
    
    def delete_prompt(self, prompt_id: int) -> Dict[str, Any]:
        """Delete a prompt."""
        response = self.client.delete(f"{API_PREFIX}/prompts/{prompt_id}")
        response.raise_for_status()
        return response.json()
    
    # Character endpoints
    def import_character(self, character_data: Dict[str, Any]) -> Dict[str, Any]:
        """Import a character card."""
        response = self.client.post(
            f"{API_PREFIX}/characters/import",
            json=character_data
        )
        response.raise_for_status()
        return response.json()
    
    def get_characters(self) -> Dict[str, Any]:
        """Get list of characters."""
        response = self.client.get(f"{API_PREFIX}/characters/")
        response.raise_for_status()
        return response.json()
    
    def delete_character(self, character_id: int) -> Dict[str, Any]:
        """Delete a character."""
        response = self.client.delete(f"{API_PREFIX}/characters/{character_id}")
        response.raise_for_status()
        return response.json()
    
    # RAG/Search endpoints
    def search_media(self, query: str, limit: int = 10) -> Dict[str, Any]:
        """Search media content."""
        response = self.client.get(
            f"{API_PREFIX}/media/search",
            params={"query": query, "limit": limit}
        )
        response.raise_for_status()
        return response.json()
    
    # Health check
    def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = self.client.get(f"{API_PREFIX}/health")
        response.raise_for_status()
        return response.json()
    
    def close(self):
        """Close the client connection."""
        self.client.close()


@pytest.fixture(scope="session")
def api_client():
    """Create an API client for the test session."""
    client = APIClient()
    # Check if single-user mode and set token
    try:
        health = client.health_check()
        if health.get("auth_mode") == "single_user":
            # Single-user mode uses the default API key or from environment
            # Try to get the actual API key from settings
            try:
                from tldw_Server_API.app.core.AuthNZ.settings import get_settings
                settings = get_settings()
                api_key = settings.SINGLE_USER_API_KEY
            except:
                api_key = os.getenv("SINGLE_USER_API_KEY", "test-api-key-12345")
            client.set_auth_token(api_key)
    except:
        pass
    yield client
    client.close()


@pytest.fixture(scope="session")
def test_user_credentials():
    """Generate test user credentials."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return {
        "username": f"e2e_test_user_{timestamp}",
        "email": f"e2e_test_{timestamp}@example.com",
        "password": "TestPassword123!"
    }


@pytest.fixture(scope="session")
def authenticated_client(api_client, test_user_credentials):
    """Create an authenticated API client."""
    # Check if single-user mode
    try:
        health = api_client.health_check()
        if health.get("auth_mode") == "single_user":
            # Single-user mode uses the default API key or from environment
            # Try to get the actual API key from settings
            try:
                from tldw_Server_API.app.core.AuthNZ.settings import get_settings
                settings = get_settings()
                api_key = settings.SINGLE_USER_API_KEY
            except:
                api_key = os.getenv("SINGLE_USER_API_KEY", "test-api-key-12345")
            api_client.set_auth_token(api_key)
            return api_client
    except:
        pass
    
    # Multi-user mode - try to register and login
    try:
        api_client.register(**test_user_credentials)
    except httpx.HTTPStatusError:
        pass  # User might already exist
    
    try:
        # Login
        api_client.login(
            test_user_credentials["username"],
            test_user_credentials["password"]
        )
    except httpx.HTTPStatusError:
        # If login fails in single-user mode, just return the client
        pass
    
    return api_client


def create_test_file(content: str, suffix: str = ".txt") -> str:
    """Create a temporary test file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=suffix, delete=False) as f:
        f.write(content)
        return f.name


def create_test_pdf() -> str:
    """Create a simple test PDF file."""
    # This would normally use a PDF library, but for testing we can use a mock
    content = b"%PDF-1.4\nTest PDF content for E2E testing"
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
        f.write(content)
        return f.name


def create_test_audio() -> str:
    """Create a simple test audio file (silent WAV)."""
    # WAV header for a silent 1-second file
    wav_header = bytes([
        0x52, 0x49, 0x46, 0x46,  # "RIFF"
        0x24, 0x08, 0x00, 0x00,  # File size
        0x57, 0x41, 0x56, 0x45,  # "WAVE"
        0x66, 0x6D, 0x74, 0x20,  # "fmt "
        0x10, 0x00, 0x00, 0x00,  # Subchunk size
        0x01, 0x00,              # Audio format (PCM)
        0x01, 0x00,              # Number of channels
        0x44, 0xAC, 0x00, 0x00,  # Sample rate (44100)
        0x88, 0x58, 0x01, 0x00,  # Byte rate
        0x02, 0x00,              # Block align
        0x10, 0x00,              # Bits per sample
        0x64, 0x61, 0x74, 0x61,  # "data"
        0x00, 0x08, 0x00, 0x00,  # Data size
    ])
    
    # Add some silence (zeros)
    silence = bytes(2048)
    
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        f.write(wav_header + silence)
        return f.name


def cleanup_test_file(file_path: str):
    """Clean up a test file."""
    try:
        os.unlink(file_path)
    except:
        pass


class TestDataTracker:
    """Track test data for cleanup."""
    
    def __init__(self):
        self.media_ids: List[int] = []
        self.note_ids: List[int] = []
        self.prompt_ids: List[int] = []
        self.character_ids: List[int] = []
        self.chat_ids: List[str] = []
        self.files: List[str] = []
        # Generic resource tracking
        from collections import defaultdict
        self.resources = defaultdict(list)
    
    def track(self, resource_type: str, resource_id: str, metadata: Dict[str, Any] = None):
        """Track a created resource for cleanup."""
        self.resources[resource_type].append({
            'id': resource_id,
            'created_at': datetime.now(),
            'metadata': metadata or {}
        })
    
    def add_media(self, media_id: int):
        self.media_ids.append(media_id)
    
    def add_note(self, note_id: int):
        self.note_ids.append(note_id)
    
    def add_prompt(self, prompt_id: int):
        self.prompt_ids.append(prompt_id)
    
    def add_character(self, character_id: int):
        self.character_ids.append(character_id)
    
    def add_chat(self, chat_id: str):
        self.chat_ids.append(chat_id)
    
    def add_file(self, file_path: str):
        self.files.append(file_path)
    
    def cleanup_files(self):
        """Clean up all tracked files."""
        for file_path in self.files:
            cleanup_test_file(file_path)


@pytest.fixture(scope="session")
def data_tracker():
    """Create a data tracker for the test session."""
    tracker = TestDataTracker()
    yield tracker
    # Cleanup files after tests
    tracker.cleanup_files()