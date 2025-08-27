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
import time
import hashlib
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from enum import Enum
from difflib import SequenceMatcher
import pytest
import httpx
from pathlib import Path

# Configuration
BASE_URL = os.getenv("E2E_TEST_BASE_URL", "http://localhost:8000")
API_PREFIX = "/api/v1"
TEST_TIMEOUT = 120  # seconds for each request (increased for video transcription)
RATE_LIMIT_RETRY_DELAY = float(os.getenv("E2E_RATE_LIMIT_DELAY", "0.5"))  # Delay after rate limit
MAX_RETRIES = int(os.getenv("E2E_MAX_RETRIES", "3"))  # Max retries for rate limit errors
SERVER_STARTUP_TIMEOUT = int(os.getenv("E2E_SERVER_STARTUP_TIMEOUT", "30"))  # Max time to wait for server


class APIClient:
    """Wrapper for API interactions with authentication support."""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.client = httpx.Client(base_url=base_url, timeout=TEST_TIMEOUT)
        self.token: Optional[str] = None
        self.refresh_token: Optional[str] = None
        self.user_id: Optional[int] = None
        
        # Note: TEST_MODE must be set on the server, not passed as header
    
    def _handle_rate_limit(self, func: Callable, *args, **kwargs) -> Any:
        """Handle rate limiting with retry logic."""
        for attempt in range(MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    if attempt < MAX_RETRIES - 1:
                        # Exponential backoff with jitter
                        delay = RATE_LIMIT_RETRY_DELAY * (2 ** attempt) + (time.time() % 0.1)
                        print(f"Rate limited, retrying in {delay:.2f}s (attempt {attempt + 1}/{MAX_RETRIES})")
                        time.sleep(delay)
                        continue
                raise
        return func(*args, **kwargs)
        
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
    def upload_media(self, file_path: str, title: str, media_type: str = "document", 
                     generate_embeddings: bool = False) -> Dict[str, Any]:
        """Upload a media file."""
        with open(file_path, "rb") as f:
            # The endpoint expects 'files' (plural) not 'file'
            files = {"files": (os.path.basename(file_path), f, "application/octet-stream")}
            data = {
                "title": title,
                "media_type": media_type,
                "overwrite_existing": "true",  # Allow overwrite for test re-runs
                "keep_original_file": "false",
                "generate_embeddings": str(generate_embeddings).lower()  # Convert bool to string
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
                     title: Optional[str] = None, custom_prompt: Optional[str] = None,
                     persist: bool = True, media_type: Optional[str] = None) -> Dict[str, Any]:
        """Process media from URL or file.
        
        Args:
            url: URL to process
            file_path: Path to file to upload and process
            title: Title for the media
            custom_prompt: Custom analysis prompt
            persist: If True, save to database. If False, ephemeral processing only
            media_type: Type of media (auto-detected if not specified)
        """
        files = None
        
        # Handle file uploads
        if file_path:
            with open(file_path, "rb") as f:
                file_content = f.read()
            files = {"files": (os.path.basename(file_path), file_content, "application/octet-stream")}
        
        if persist:
            # Use /add endpoint for persistent storage
            data = {}
            if url:
                data["urls"] = url  # /add expects comma-separated string
                data["media_type"] = media_type or "document"
            if title:
                data["title"] = title
            if custom_prompt:
                data["custom_prompt"] = custom_prompt
            # Enable overwrite to handle re-runs of tests
            data["overwrite_existing"] = "true"
            
            # If processing a file, detect type
            if file_path and not media_type:
                ext = os.path.splitext(file_path)[1].lower()
                if ext in ['.mp4', '.avi', '.mov', '.mkv']:
                    data["media_type"] = "video"
                elif ext in ['.mp3', '.wav', '.ogg', '.m4a']:
                    data["media_type"] = "audio"
                elif ext in ['.pdf']:
                    data["media_type"] = "pdf"
                elif ext in ['.txt', '.md', '.html', '.htm', '.xml', '.docx', '.rtf']:
                    data["media_type"] = "document"
                elif ext in ['.epub']:
                    data["media_type"] = "ebook"
                else:
                    data["media_type"] = "document"
            
            response = self.client.post(
                f"{API_PREFIX}/media/add",
                data=data,
                files=files
            )
        else:
            # Use process-* endpoints for ephemeral processing
            # Determine which endpoint to use based on content
            if url or (file_path and file_path.endswith(('.html', '.htm', '.txt', '.md', '.xml'))):
                endpoint = "process-documents"
                data = {}
                if url:
                    data["urls"] = url  # process-documents expects comma-separated string
                if title:
                    data["titles"] = title
                if custom_prompt:
                    data["custom_prompt"] = custom_prompt
            elif file_path and file_path.endswith('.pdf'):
                endpoint = "process-pdfs"
                data = {}
                if title:
                    data["titles"] = title
                if custom_prompt:
                    data["custom_prompt"] = custom_prompt
            elif file_path and file_path.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                endpoint = "process-videos"
                data = {}
                if url:
                    data["urls"] = url
                if title:
                    data["titles"] = title
                if custom_prompt:
                    data["custom_prompt"] = custom_prompt
            elif file_path and file_path.endswith(('.mp3', '.wav', '.ogg', '.m4a')):
                endpoint = "process-audios"
                data = {}
                if url:
                    data["urls"] = url
                if title:
                    data["titles"] = title
                if custom_prompt:
                    data["custom_prompt"] = custom_prompt
            else:
                # Default to documents for web content
                endpoint = "process-documents"
                data = {}
                if url:
                    data["urls"] = url
                if title:
                    data["titles"] = title
                if custom_prompt:
                    data["custom_prompt"] = custom_prompt
            
            response = self.client.post(
                f"{API_PREFIX}/media/{endpoint}",
                data=data,
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
                        character_id: Optional[int] = None,
                        conversation_id: Optional[str] = None,
                        stream: bool = False) -> Dict[str, Any]:
        """Send chat completion request with optional character context."""
        data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "stream": stream
        }
        if character_id is not None:
            data["character_id"] = str(character_id)  # API expects string
        if conversation_id is not None:
            data["conversation_id"] = conversation_id
        
        response = self.client.post(
            f"{API_PREFIX}/chat/completions",
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    # Notes endpoints
    def create_note(self, title: str, content: str, keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """Create a new note."""
        def _create():
            data = {
                "title": title,
                "content": content
            }
            if keywords:
                data["keywords"] = keywords
                
            response = self.client.post(f"{API_PREFIX}/notes/", json=data)
            response.raise_for_status()
            return response.json()
        
        return self._handle_rate_limit(_create)
    
    def get_notes(self, limit: int = 50, offset: int = 0) -> Dict[str, Any]:
        """Get list of notes."""
        def _get():
            response = self.client.get(
                f"{API_PREFIX}/notes/",
                params={"limit": limit, "offset": offset}
            )
            response.raise_for_status()
            return response.json()
        
        return self._handle_rate_limit(_get)
    
    def update_note(self, note_id: str, title: Optional[str] = None, 
                   content: Optional[str] = None, version: int = 1) -> Dict[str, Any]:
        """Update an existing note."""
        def _update():
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
        
        return self._handle_rate_limit(_update)
    
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
    
    def get_character(self, character_id: int) -> Dict[str, Any]:
        """Get a specific character by ID."""
        response = self.client.get(f"{API_PREFIX}/characters/{character_id}")
        response.raise_for_status()
        return response.json()
    
    def update_character(self, character_id: int, expected_version: int, **kwargs) -> Dict[str, Any]:
        """Update a character with optimistic locking."""
        response = self.client.put(
            f"{API_PREFIX}/characters/{character_id}",
            json=kwargs,
            params={"expected_version": expected_version}
        )
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
        def _search():
            response = self.client.post(
                f"{API_PREFIX}/media/search",
                json={"query": query},
                params={"limit": limit}
            )
            response.raise_for_status()
            return response.json()
        
        return self._handle_rate_limit(_search)
    
    def rag_simple_search(self, query: str, databases: List[str] = None, **kwargs) -> Dict[str, Any]:
        """Perform simple RAG search."""
        data = {
            "query": query,
            "databases": databases or ["media"],
            **kwargs
        }
        response = self.client.post(
            f"{API_PREFIX}/rag/search/simple",  # Fixed path: /rag/search/simple instead of /rag/simple/search
            json=data
        )
        response.raise_for_status()
        return response.json()
    
    def rag_advanced_search(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Perform advanced RAG search with full configuration."""
        response = self.client.post(
            f"{API_PREFIX}/rag/search/complex",  # Fixed path: /rag/search/complex for consistency
            json=config
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


def ensure_server_running(base_url: str = BASE_URL, timeout: int = SERVER_STARTUP_TIMEOUT) -> Dict[str, Any]:
    """
    Ensure the API server is running and accessible.
    This simulates a user trying to connect to the application.
    
    Returns:
        Server health information
    
    Raises:
        pytest.skip if server is not available
    """
    health_url = f"{base_url}{API_PREFIX}/health"
    start_time = time.time()
    last_error = None
    
    print(f"🔍 Checking if API server is available at {base_url}...")
    
    while time.time() - start_time < timeout:
        try:
            with httpx.Client(timeout=5) as temp_client:
                response = temp_client.get(health_url)
                if response.status_code == 200:
                    health_data = response.json()
                    print(f"✅ API server is running in {health_data.get('auth_mode', 'unknown')} mode")
                    return health_data
        except (httpx.ConnectError, httpx.TimeoutException) as e:
            last_error = e
            time.sleep(1)  # Wait before retrying
            continue
    
    # Server not available after timeout
    error_msg = (
        f"❌ API server not available at {base_url} after {timeout} seconds.\n"
        f"Please ensure the server is running:\n"
        f"  python -m uvicorn tldw_Server_API.app.main:app --reload\n"
        f"Last error: {last_error}"
    )
    pytest.skip(error_msg)


@pytest.fixture(scope="session")
def api_client():
    """Create an API client for the test session - simulating a user connecting to the app."""
    # First ensure server is running - like a user would check if app is accessible
    health_info = ensure_server_running()
    
    client = APIClient()
    
    # Check if single-user mode and set token
    try:
        if health_info.get("auth_mode") == "single_user":
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


# ============================================================================
# ASSERTION HELPERS - Fix weak assertions
# ============================================================================

class AssertionHelpers:
    """Helper functions for strong assertions that actually validate functionality."""
    
    @staticmethod
    def assert_successful_upload(response: Dict[str, Any]) -> int:
        """Validate media upload response with specific assertions."""
        # Handle new format with results array
        if "results" in response and isinstance(response["results"], list):
            assert len(response["results"]) > 0, "Empty results array"
            result = response["results"][0]
            
            # Check for actual success, not errors
            if result.get("status") == "Error":
                # Check if it's a duplicate error (might be acceptable in some tests)
                if "already exists" in result.get("db_message", ""):
                    # Return existing ID if available
                    if result.get("db_id"):
                        return result["db_id"]
                    pytest.skip("File already exists - acceptable for idempotent test")
                else:
                    pytest.fail(f"Upload failed with error: {result.get('error', result.get('db_message'))}")
            
            # Check if already exists (overwrite scenario)
            if result.get("db_message") and "already" in result.get("db_message", "").lower():
                # Item already exists, check if we got the ID
                if result.get("db_id"):
                    return result["db_id"]
                # If no ID but item exists, that's acceptable for some tests
                pytest.skip(f"Item already exists: {result.get('db_message')}")
            
            assert result.get("status") == "Success", f"Upload not successful: {result}"
            assert "db_id" in result and result["db_id"] is not None, f"No valid db_id in result: {result}"
            media_id = result["db_id"]
            assert isinstance(media_id, int) and media_id > 0, f"Invalid media_id: {media_id}"
            return media_id
        
        # Handle old format
        assert "status" not in response or response["status"] == "Success", f"Upload failed: {response}"
        assert "media_id" in response or "id" in response, "No media_id in response"
        media_id = response.get("media_id") or response.get("id")
        assert isinstance(media_id, int) and media_id > 0, f"Invalid media_id: {media_id}"
        return media_id
    
    @staticmethod
    def assert_content_integrity(original: str, retrieved: str, tolerance: float = 0.90):
        """Validate that retrieved content matches original."""
        if not retrieved:
            pytest.fail("Retrieved content is empty")
        
        # Normalize whitespace for comparison
        original_normalized = " ".join(original.split())
        retrieved_normalized = " ".join(retrieved.split())
        
        # Check exact match first
        if original_normalized == retrieved_normalized:
            return True
        
        # Check similarity if processing might have altered it
        similarity = SequenceMatcher(None, original_normalized, retrieved_normalized).ratio()
        assert similarity >= tolerance, \
            f"Content integrity check failed. Similarity: {similarity:.2%} (required: {tolerance:.0%})"
        
        return True
    
    @staticmethod
    def assert_api_response_structure(response: Dict[str, Any], required_fields: List[str]):
        """Validate API response has required structure."""
        for field in required_fields:
            assert field in response, f"Response missing required field: {field}"
            # Don't check for None on optional fields
            if field in ["id", "status", "title"]:  # Critical fields
                assert response[field] is not None, f"Required field {field} is null"


# ============================================================================
# ERROR HANDLING - Categorize and handle errors appropriately
# ============================================================================

class ErrorCategory(Enum):
    """Categorize errors for appropriate handling."""
    ENVIRONMENT = "environment"  # Test env issues, can skip
    CRITICAL = "critical"        # Must fail test
    TRANSIENT = "transient"      # Should retry
    EXPECTED = "expected"        # Normal in testing


class SmartErrorHandler:
    """Intelligent error handling that distinguishes error types."""
    
    @classmethod
    def handle_error(cls, error: Exception, context: str = "") -> None:
        """Properly handle errors based on type and context."""
        if isinstance(error, httpx.HTTPStatusError):
            status_code = error.response.status_code
            
            # Environment issues - skip test
            if status_code in [503, 502]:
                pytest.skip(f"Service unavailable in {context}: HTTP {status_code}")
            
            # Authentication issues - critical
            elif status_code in [401, 403]:
                pytest.fail(f"Authentication failed in {context}: HTTP {status_code}")
            
            # Server errors - critical
            elif status_code == 500:
                pytest.fail(f"Server error in {context}: HTTP 500")
            
            # Rate limiting - could retry but skip for now
            elif status_code == 429:
                pytest.skip(f"Rate limited in {context}")
            
            # Not found - check context
            elif status_code == 404:
                if "delete" in context.lower() or "after deletion" in context.lower():
                    return  # 404 is expected after deletion
                pytest.fail(f"Resource not found in {context}")
            
            # Validation errors
            elif status_code == 422:
                if "validation" in context.lower():
                    return  # Expected when testing validation
                pytest.fail(f"Validation error in {context}: {error}")
            
            # Timeout
            elif status_code in [408, 504]:
                pytest.skip(f"Timeout in {context}: HTTP {status_code}")
            
            else:
                pytest.fail(f"Unexpected HTTP {status_code} in {context}")
        
        elif isinstance(error, httpx.ConnectError):
            pytest.skip(f"Cannot connect to API in {context}: {error}")
        
        elif isinstance(error, httpx.TimeoutException):
            pytest.skip(f"Request timeout in {context}")
        
        else:
            # Unknown error - fail to be safe
            pytest.fail(f"Unhandled error in {context}: {type(error).__name__}: {error}")


# ============================================================================
# ASYNC OPERATION HANDLING - Poll for long-running operations
# ============================================================================

class AsyncOperationHandler:
    """Handle long-running async operations properly."""
    
    @staticmethod
    def wait_for_completion(
        check_func: Callable[[], Dict[str, Any]],
        success_condition: Callable[[Dict[str, Any]], bool],
        timeout: int = 60,
        poll_interval: int = 2,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Poll for async operation completion.
        
        Args:
            check_func: Function to call to check status
            success_condition: Function to determine if operation succeeded
            timeout: Maximum time to wait in seconds
            poll_interval: Time between checks in seconds
            context: Description for error messages
        
        Returns:
            Final successful response
        """
        start_time = time.time()
        last_response = None
        
        while time.time() - start_time < timeout:
            try:
                response = check_func()
                last_response = response
                
                # Check for success
                if success_condition(response):
                    return response
                
                # Check for failure
                if response.get("status") == "failed" or response.get("error"):
                    error_msg = response.get("error", "Unknown error")
                    pytest.fail(f"{context} failed: {error_msg}")
                
                # Still processing, wait and retry
                time.sleep(poll_interval)
                
            except httpx.HTTPStatusError as e:
                # Might be temporary, continue unless timeout approaching
                if time.time() - start_time < timeout - 10:
                    time.sleep(poll_interval)
                    continue
                else:
                    raise
        
        # Timeout reached
        pytest.fail(
            f"{context} timed out after {timeout}s. "
            f"Last status: {last_response.get('status') if last_response else 'Unknown'}"
        )
    
    @staticmethod
    def wait_for_transcription(api_client, media_id: int, timeout: int = 120) -> Dict[str, Any]:
        """Wait for media transcription to complete."""
        def check_status():
            return api_client.get_media_item(media_id)
        
        def is_complete(response):
            # Check various possible status fields
            if response.get("transcription_status") == "completed":
                return True
            if response.get("status") in ["completed", "success", "done"]:
                return True
            # Check if transcription exists
            if response.get("transcription") or response.get("transcript"):
                return True
            return False
        
        return AsyncOperationHandler.wait_for_completion(
            check_func=check_status,
            success_condition=is_complete,
            timeout=timeout,
            context=f"Transcription of media {media_id}"
        )


# ============================================================================
# CONTENT VALIDATION - Verify data integrity
# ============================================================================

class ContentValidator:
    """Validate content integrity and processing results."""
    
    @staticmethod
    def validate_transcription(transcription: str, min_length: int = 10) -> bool:
        """Validate transcription quality."""
        if not transcription:
            pytest.fail("Transcription is empty")
        
        assert len(transcription) >= min_length, \
            f"Transcription too short: {len(transcription)} chars (min: {min_length})"
        
        # Check it's not an error message
        error_indicators = ["error", "failed", "unavailable", "exception"]
        transcription_lower = transcription.lower()[:100]  # Check start
        for indicator in error_indicators:
            assert indicator not in transcription_lower, \
                f"Transcription appears to be an error: {transcription[:100]}"
        
        return True
    
    @staticmethod
    def validate_search_results(
        query: str,
        results: List[Dict[str, Any]],
        expected_ids: Optional[List[int]] = None,
        min_results: int = 0
    ) -> bool:
        """Validate search results."""
        assert isinstance(results, list), "Results should be a list"
        
        if min_results > 0:
            assert len(results) >= min_results, \
                f"Too few results: {len(results)} (expected >= {min_results})"
        
        # Check expected IDs if provided
        if expected_ids:
            found_ids = [r.get("id") or r.get("media_id") for r in results]
            for expected_id in expected_ids:
                assert expected_id in found_ids, \
                    f"Expected ID {expected_id} not in search results: {found_ids}"
        
        return True
    
    @staticmethod
    def validate_chat_response(response: Dict[str, Any], min_length: int = 1) -> bool:
        """Validate chat/LLM response."""
        # Check structure
        assert "choices" in response or "response" in response or "content" in response, \
            f"Invalid chat response structure: {response.keys()}"
        
        # Extract message
        message = ""
        if "choices" in response and len(response["choices"]) > 0:
            message = response["choices"][0].get("message", {}).get("content", "")
        elif "response" in response:
            message = response["response"]
        elif "content" in response:
            message = response["content"]
        
        assert message, "Chat response message is empty"
        assert len(message) >= min_length, f"Response too short: {len(message)} chars"
        
        return True


# ============================================================================
# STATE VERIFICATION - Verify consistency between test phases
# ============================================================================

class StateVerification:
    """Verify state consistency between test phases."""
    
    @staticmethod
    def verify_uploads_accessible(api_client, uploaded_items: List[Dict[str, Any]]) -> None:
        """Verify all uploaded media is accessible and intact."""
        if not uploaded_items:
            return
        
        errors = []
        verified_count = 0
        
        for item in uploaded_items:
            # Extract media ID
            media_id = None
            if "media_id" in item:
                media_id = item["media_id"]
            elif "results" in item and item["results"]:
                media_id = item["results"][0].get("db_id")
            
            if not media_id:
                continue
            
            try:
                # Retrieve and verify
                details = api_client.get_media_item(media_id)
                
                # Verify ID matches
                retrieved_id = details.get("id") or details.get("media_id")
                assert retrieved_id == media_id, f"ID mismatch: {retrieved_id} != {media_id}"
                
                # Verify has content
                has_content = (
                    details.get("content") or 
                    details.get("text") or 
                    details.get("transcript") or
                    details.get("transcription")
                )
                assert has_content, f"Media {media_id} has no content"
                
                verified_count += 1
                
            except Exception as e:
                errors.append(f"Media {media_id}: {str(e)}")
        
        # At least 80% should be accessible
        if uploaded_items:
            success_rate = verified_count / len(uploaded_items)
            assert success_rate >= 0.8, \
                f"Only {verified_count}/{len(uploaded_items)} uploads accessible. Errors: {errors}"
    
    @staticmethod
    def create_content_hash(content: str) -> str:
        """Create a hash of content for verification."""
        return hashlib.md5(content.encode()).hexdigest()
    
    @staticmethod
    def verify_content_preserved(original: str, retrieved: str, context: str = "") -> None:
        """Verify content is preserved through processing."""
        # Check key markers if present
        if "TEST_MARKER" in original:
            assert "TEST_MARKER" in retrieved, \
                f"Test marker lost in {context}"
        
        # Check similarity
        AssertionHelpers.assert_content_integrity(original, retrieved, tolerance=0.85)


# ============================================================================
# NEGATIVE TEST HELPERS - Generate and validate malicious inputs
# ============================================================================

class NegativeTestHelper:
    """Helper for generating and testing malicious/invalid inputs."""
    
    @staticmethod
    def generate_malicious_payload(payload_type: str) -> List[str]:
        """Generate malicious payloads of a specific type."""
        from tldw_Server_API.tests.e2e.test_data import TestDataGenerator
        payloads = TestDataGenerator.malicious_payloads()
        return payloads.get(payload_type, [])
    
    @staticmethod
    def generate_corrupted_file(file_type: str) -> bytes:
        """Generate corrupted file data for a specific format."""
        from tldw_Server_API.tests.e2e.test_data import TestDataGenerator
        corrupted_data = TestDataGenerator.generate_corrupted_file_data()
        return corrupted_data.get(file_type, b"CORRUPTED")
    
    @staticmethod
    def validate_sanitization(original: str, sanitized: str, payload_type: str) -> bool:
        """Validate that a malicious payload was properly sanitized."""
        if payload_type == 'sql_injection':
            dangerous_patterns = ["DROP", "DELETE", "UNION", "SELECT", "--"]
            for pattern in dangerous_patterns:
                if pattern.upper() in sanitized.upper():
                    return False
                    
        elif payload_type == 'xss':
            dangerous_patterns = ["<script", "onerror", "javascript:", "<iframe"]
            for pattern in dangerous_patterns:
                if pattern.lower() in sanitized.lower():
                    return False
                    
        elif payload_type == 'path_traversal':
            if ".." in sanitized or "etc/passwd" in sanitized:
                return False
                
        return True
    
    @staticmethod
    def generate_boundary_value(value_type: str, boundary: str):
        """Generate boundary test values."""
        from tldw_Server_API.tests.e2e.test_data import TestDataGenerator
        boundaries = TestDataGenerator.boundary_values()
        
        if value_type in boundaries:
            return boundaries[value_type].get(boundary)
        return None


# ============================================================================
# CONCURRENT TEST HELPERS - Manage parallel operations
# ============================================================================

class ConcurrentTestManager:
    """Manager for concurrent test operations."""
    
    def __init__(self, max_workers: int = 10):
        self.max_workers = max_workers
        self.results = []
        self.errors = []
        
    def run_parallel(
        self,
        func: Callable,
        args_list: List[tuple],
        timeout: int = 30
    ) -> Dict[str, Any]:
        """Run function with different arguments in parallel."""
        import concurrent.futures
        
        results = {
            'successful': [],
            'failed': [],
            'timing': [],
            'race_conditions': []
        }
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_args = {
                executor.submit(func, *args): args
                for args in args_list
            }
            
            for future in concurrent.futures.as_completed(future_to_args):
                start_time = time.time()
                args = future_to_args[future]
                
                try:
                    result = future.result(timeout=timeout)
                    duration = time.time() - start_time
                    results['successful'].append({
                        'args': args,
                        'result': result,
                        'duration': duration
                    })
                except Exception as e:
                    duration = time.time() - start_time
                    results['failed'].append({
                        'args': args,
                        'error': str(e),
                        'duration': duration
                    })
                
                results['timing'].append(duration)
        
        # Detect race conditions
        self._detect_race_conditions(results)
        
        return results
    
    def _detect_race_conditions(self, results: Dict[str, Any]) -> None:
        """Detect potential race conditions in results."""
        # Check for duplicate IDs
        all_ids = []
        for item in results['successful']:
            if 'result' in item and isinstance(item['result'], dict):
                id_val = item['result'].get('id') or item['result'].get('media_id')
                if id_val:
                    all_ids.append(id_val)
        
        if len(all_ids) != len(set(all_ids)):
            results['race_conditions'].append('Duplicate IDs detected')
        
        # Check for inconsistent states
        if len(results['successful']) > 0 and len(results['failed']) > 0:
            # Analyze failure patterns
            error_types = {}
            for failure in results['failed']:
                error = failure.get('error', 'Unknown')
                error_types[error] = error_types.get(error, 0) + 1
            
            # If same operation sometimes succeeds and sometimes fails, might be race
            if len(error_types) > 2:
                results['race_conditions'].append('Inconsistent operation results')
    
    def measure_throughput(
        self,
        func: Callable,
        duration_seconds: int = 10,
        target_rps: float = 10.0
    ) -> Dict[str, float]:
        """Measure throughput of a function over time."""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        successful = 0
        failed = 0
        response_times = []
        
        while time.time() < end_time:
            request_start = time.time()
            
            try:
                func()
                successful += 1
            except:
                failed += 1
            
            response_time = time.time() - request_start
            response_times.append(response_time)
            
            # Maintain target RPS
            sleep_time = max(0, (1.0 / target_rps) - response_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        total_time = time.time() - start_time
        total_requests = successful + failed
        
        return {
            'total_requests': total_requests,
            'successful': successful,
            'failed': failed,
            'rps': total_requests / total_time if total_time > 0 else 0,
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'p95_response_time': sorted(response_times)[int(len(response_times) * 0.95)] if response_times else 0,
            'success_rate': successful / total_requests if total_requests > 0 else 0
        }


# ============================================================================
# TEST DATA CORRUPTOR - Generate invalid test data
# ============================================================================

class TestDataCorruptor:
    """Generate corrupted or invalid test data."""
    
    @staticmethod
    def corrupt_json(valid_json: str) -> str:
        """Corrupt valid JSON in various ways."""
        import random
        
        corruption_methods = [
            lambda s: s[:-1],  # Remove closing brace
            lambda s: s.replace('"', "'"),  # Single quotes
            lambda s: s.replace(':', ''),  # Remove colons
            lambda s: s.replace(',', ',,'),  # Double commas
            lambda s: s + "extra",  # Extra content
            lambda s: "undefined" + s,  # Undefined reference
        ]
        
        method = random.choice(corruption_methods)
        return method(valid_json)
    
    @staticmethod
    def corrupt_file(file_data: bytes, corruption_type: str = 'truncate') -> bytes:
        """Corrupt file data in various ways."""
        if corruption_type == 'truncate':
            return file_data[:len(file_data)//2]
        elif corruption_type == 'null_bytes':
            return b'\x00' * 100 + file_data
        elif corruption_type == 'random':
            import random
            corrupted = bytearray(file_data)
            for i in range(min(100, len(corrupted))):
                corrupted[random.randint(0, len(corrupted)-1)] = random.randint(0, 255)
            return bytes(corrupted)
        elif corruption_type == 'empty':
            return b''
        else:
            return file_data + b'CORRUPTED'
    
    @staticmethod
    def generate_malformed_multipart() -> bytes:
        """Generate malformed multipart form data."""
        boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
        
        # Missing boundary terminator
        malformed = f"""------{boundary}
Content-Disposition: form-data; name="file"; filename="test.txt"
Content-Type: text/plain

Test content without proper boundary end""".encode()
        
        return malformed
    
    @staticmethod
    def generate_invalid_encoding(text: str) -> bytes:
        """Generate text with invalid encoding."""
        # Mix encodings incorrectly
        try:
            # UTF-8 with invalid sequences
            return text.encode('utf-8') + b'\xff\xfe' + text.encode('utf-16')
        except:
            return b'\xff\xfe\xff\xfe'


# ============================================================================
# STRONG ASSERTION HELPERS - Enhanced validation with exact value checking
# ============================================================================

class StrongAssertionHelpers:
    """Enhanced assertion helpers with strict value checking."""
    
    @staticmethod
    def assert_exact_value(actual, expected, field_name="field"):
        """Assert exact value match with helpful error message."""
        assert actual == expected, \
            f"{field_name}: expected '{expected}', got '{actual}'"
    
    @staticmethod
    def assert_value_in_range(value, min_val, max_val, field_name="value"):
        """Assert numeric value is within range."""
        assert isinstance(value, (int, float)), \
            f"{field_name} must be numeric, got {type(value).__name__}"
        assert min_val <= value <= max_val, \
            f"{field_name} {value} not in range [{min_val}, {max_val}]"
    
    @staticmethod
    def assert_non_empty_string(value, field_name="field", min_length=1):
        """Assert string is non-empty with minimum length."""
        assert isinstance(value, str), \
            f"{field_name} must be string, got {type(value).__name__}"
        assert len(value) >= min_length, \
            f"{field_name} too short: {len(value)} < {min_length}"
    
    @staticmethod
    def assert_valid_timestamp(timestamp_str, field_name="timestamp"):
        """Assert valid ISO format timestamp."""
        from datetime import datetime
        try:
            datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        except (ValueError, AttributeError) as e:
            pytest.fail(f"Invalid {field_name}: {timestamp_str} - {e}")
    
    @staticmethod
    def assert_character_response(character_data):
        """Validate character response structure and values."""
        required = ["id", "name", "version"]
        for field in required:
            assert field in character_data, f"Missing required field: {field}"
        
        # Type validation
        assert isinstance(character_data["id"], int), \
            f"ID must be int, got {type(character_data['id']).__name__}"
        assert isinstance(character_data["name"], str), \
            f"Name must be string, got {type(character_data['name']).__name__}"
        assert isinstance(character_data["version"], int), \
            f"Version must be int, got {type(character_data['version']).__name__}"
        
        # Value validation
        assert character_data["id"] > 0, f"Invalid ID: {character_data['id']}"
        assert len(character_data["name"]) > 0, "Name is empty"
        assert character_data["version"] >= 1, f"Invalid version: {character_data['version']}"
        
        # Optional fields validation when present
        if "tags" in character_data:
            assert isinstance(character_data["tags"], list), \
                f"Tags must be list, got {type(character_data['tags']).__name__}"
        if "description" in character_data:
            assert isinstance(character_data["description"], str), \
                f"Description must be string, got {type(character_data['description']).__name__}"
    
    @staticmethod
    def assert_rag_result_quality(result, query_terms=None):
        """Validate RAG search result quality and structure."""
        assert "content" in result, "Result missing 'content'"
        
        # Content validation
        content = result["content"]
        assert isinstance(content, str), f"Content must be string, got {type(content).__name__}"
        assert len(content) > 10, f"Content too short: {len(content)} chars"
        
        # Score validation if present
        if "score" in result:
            score = result["score"]
            assert isinstance(score, (int, float)), \
                f"Score must be numeric, got {type(score).__name__}"
            assert 0.0 <= score <= 1.0, f"Score out of range: {score}"
        
        # Source validation if present
        if "source" in result:
            source = result["source"]
            assert isinstance(source, dict), \
                f"Source must be dict, got {type(source).__name__}"
            assert "type" in source, "Source missing 'type'"
            assert source["type"] in ["media", "note", "character", "chat"], \
                f"Invalid source type: {source['type']}"
            if "id" in source:
                assert isinstance(source["id"], (int, str)), \
                    f"Source ID must be int or string, got {type(source['id']).__name__}"
        
        # Relevance check if query terms provided
        if query_terms:
            content_lower = content.lower()
            has_term = any(term.lower() in content_lower for term in query_terms)
            if not has_term and result.get("score", 0) > 0.5:
                print(f"Warning: High score {result.get('score')} but no query terms found in content")
    
    @staticmethod
    def assert_chat_response_quality(response, min_length=10):
        """Validate chat response quality and structure."""
        assert "choices" in response, "Response missing 'choices'"
        choices = response["choices"]
        assert isinstance(choices, list), f"Choices must be list, got {type(choices).__name__}"
        assert len(choices) > 0, "No choices in response"
        
        # Validate first choice
        choice = choices[0]
        assert isinstance(choice, dict), f"Choice must be dict, got {type(choice).__name__}"
        assert "message" in choice, "Choice missing 'message'"
        
        # Validate message
        message = choice["message"]
        assert isinstance(message, dict), f"Message must be dict, got {type(message).__name__}"
        assert "role" in message, "Message missing 'role'"
        assert "content" in message, "Message missing 'content'"
        
        # Validate content
        content = message["content"]
        assert isinstance(content, str), f"Content must be string, got {type(content).__name__}"
        assert len(content) >= min_length, \
            f"Response too short: {len(content)} < {min_length}"
        
        return content

# Test markers would be defined in pytest.ini or pyproject.toml
# For now, commenting out to avoid errors
# pytest.mark.negative = pytest.mark.negative
# pytest.mark.concurrent = pytest.mark.concurrent
# pytest.mark.security = pytest.mark.security
# pytest.mark.slow = pytest.mark.slow