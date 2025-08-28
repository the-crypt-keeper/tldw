"""
Workflow Test Helpers
----------------------

Helper classes and utilities specifically designed for the sequential workflow
testing pattern used in the E2E tests.
"""

import pytest
import httpx
from typing import Dict, Any, Optional, List
from difflib import SequenceMatcher


class WorkflowAssertions:
    """Assertions that validate responses while preserving workflow data."""
    
    @staticmethod
    def assert_valid_upload(response: Dict[str, Any], expected_title: Optional[str] = None) -> int:
        """
        Validate upload response and return media_id for workflow continuation.
        
        Args:
            response: The upload response from API
            expected_title: Optional title to verify
            
        Returns:
            media_id for use in subsequent workflow steps
        """
        assert response is not None, "Upload response is None"
        
        # Handle both direct response and results array format
        if "results" in response and isinstance(response["results"], list):
            assert len(response["results"]) > 0, "Empty results array in upload response"
            result = response["results"][0]
            
            # Check for actual success
            if result.get("status") == "Error":
                # Check if it's a duplicate (might be acceptable)
                if "already exists" in result.get("db_message", ""):
                    if result.get("db_id"):
                        return result["db_id"]  # Return existing ID
                    pytest.fail(f"Upload failed - file exists but no ID returned: {result}")
                else:
                    pytest.fail(f"Upload failed with error: {result.get('error', result.get('db_message'))}")
            
            media_id = result.get("db_id")
        else:
            # Old format compatibility
            media_id = response.get("media_id") or response.get("id")
        
        # Validate media_id
        assert media_id is not None, f"No media_id in upload response: {response}"
        assert isinstance(media_id, int), f"Invalid media_id type: {type(media_id)}, value: {media_id}"
        assert media_id > 0, f"Invalid media_id value: {media_id}"
        
        # Verify title if provided
        if expected_title:
            if "title" in response:
                assert response["title"] == expected_title, f"Title mismatch: expected '{expected_title}', got '{response['title']}'"
        
        return media_id
    
    @staticmethod
    def assert_valid_note(response: Dict[str, Any]) -> int:
        """Validate note creation/update response."""
        assert response is not None, "Note response is None"
        
        # Check for note ID
        note_id = response.get("id") or response.get("note_id")
        assert note_id is not None, f"No note_id in response: {response}"
        
        # If it's a string ID (some systems use UUIDs)
        if isinstance(note_id, str):
            assert len(note_id) > 0, "Note ID is empty string"
        else:
            assert isinstance(note_id, int), f"Invalid note_id type: {type(note_id)}"
            assert note_id > 0, f"Invalid note_id value: {note_id}"
        
        return note_id
    
    @staticmethod
    def assert_valid_chat_response(response: Dict[str, Any], min_length: int = 1) -> str:
        """
        Validate chat response and return the message content.
        
        Returns:
            The actual message content for verification
        """
        assert response is not None, "Chat response is None"
        
        # Check structure
        assert "choices" in response or "response" in response or "content" in response, \
            f"Invalid chat response structure. Keys: {response.keys()}"
        
        # Extract message
        message = ""
        if "choices" in response and isinstance(response["choices"], list) and len(response["choices"]) > 0:
            choice = response["choices"][0]
            assert "message" in choice, f"No message in choice: {choice}"
            message = choice["message"].get("content", "")
        elif "response" in response:
            message = response["response"]
        elif "content" in response:
            message = response["content"]
        
        # Validate message
        assert message, "Chat response message is empty"
        assert isinstance(message, str), f"Message should be string, got: {type(message)}"
        assert len(message) >= min_length, f"Response too short: {len(message)} chars (min: {min_length})"
        
        return message


class WorkflowErrorHandler:
    """Handle errors appropriately for workflow testing."""
    
    @staticmethod
    def handle_api_error(error: Exception, operation: str, skip_on_not_implemented: bool = True):
        """
        Handle API errors with proper categorization.
        
        Args:
            error: The exception that occurred
            operation: Description of the operation that failed
            skip_on_not_implemented: Whether to skip on 501 errors
        """
        if isinstance(error, httpx.HTTPStatusError):
            status = error.response.status_code
            response_text = error.response.text[:500]  # First 500 chars of response
            
            # Feature not implemented - skip if allowed
            if status == 501 and skip_on_not_implemented:
                pytest.skip(f"{operation} not implemented (501)")
            
            # Authentication issues in single-user mode - skip
            elif status in [401, 403]:
                if "single" in response_text.lower() or "single_user" in response_text.lower():
                    pytest.skip(f"{operation} not available in single-user mode")
                else:
                    pytest.fail(f"Authentication failed for {operation}: HTTP {status}")
            
            # Not found - might be expected
            elif status == 404:
                if "endpoint" in response_text.lower():
                    pytest.skip(f"{operation} endpoint not found (404)")
                else:
                    pytest.fail(f"Resource not found in {operation}: HTTP 404")
            
            # Bad request - likely test issue
            elif status == 400:
                pytest.fail(f"Bad request in {operation}: {response_text}")
            
            # Server errors - always fail
            elif status >= 500:
                pytest.fail(f"Server error in {operation}: HTTP {status} - {response_text}")
            
            # Validation errors
            elif status == 422:
                pytest.fail(f"Validation error in {operation}: {response_text}")
            
            # Other client errors
            elif status >= 400:
                pytest.fail(f"Client error in {operation}: HTTP {status}")
            
            # Unexpected status
            else:
                pytest.fail(f"Unexpected status {status} in {operation}")
        
        elif isinstance(error, httpx.ConnectError):
            pytest.fail(f"Cannot connect to API for {operation}: {error}")
        
        elif isinstance(error, httpx.TimeoutException):
            pytest.fail(f"Request timeout in {operation}: {error}")
        
        else:
            # Unknown error type - fail test
            pytest.fail(f"Unexpected error in {operation}: {type(error).__name__}: {error}")
    
    @staticmethod
    def is_recoverable_error(error: Exception) -> bool:
        """Check if an error is recoverable (should retry)."""
        if isinstance(error, httpx.HTTPStatusError):
            # Retry on rate limiting or temporary server errors
            return error.response.status_code in [429, 502, 503, 504]
        elif isinstance(error, httpx.TimeoutException):
            return True  # Timeouts might be temporary
        return False


class WorkflowVerification:
    """Verification utilities for workflow checkpoints."""
    
    @staticmethod
    def verify_phase_transition(
        previous_phase_data: List[Dict],
        required_success_rate: float = 0.8
    ) -> bool:
        """
        Verify that enough operations from the previous phase succeeded.
        
        Args:
            previous_phase_data: List of items from previous phase
            required_success_rate: Minimum success rate to continue
            
        Returns:
            True if phase transition is safe
        """
        if not previous_phase_data:
            pytest.skip("No data from previous phase")
            return False
        
        successful = sum(1 for item in previous_phase_data if item and item.get("id") or item.get("media_id"))
        total = len(previous_phase_data)
        success_rate = successful / total if total > 0 else 0
        
        if success_rate < required_success_rate:
            pytest.fail(
                f"Previous phase success rate too low: {success_rate:.0%} "
                f"(required: {required_success_rate:.0%}). "
                f"Successful: {successful}/{total}"
            )
        
        return True
    
    @staticmethod
    def verify_content_similarity(
        original: str,
        retrieved: str,
        min_similarity: float = 0.85
    ) -> bool:
        """
        Verify content similarity (processing might alter formatting).
        
        Args:
            original: Original content
            retrieved: Retrieved/processed content
            min_similarity: Minimum acceptable similarity ratio
            
        Returns:
            True if content is similar enough
        """
        if not retrieved:
            return False
        
        # Normalize whitespace for comparison
        original_normalized = " ".join(original.split())
        retrieved_normalized = " ".join(retrieved.split())
        
        # Calculate similarity
        similarity = SequenceMatcher(None, original_normalized, retrieved_normalized).ratio()
        
        if similarity < min_similarity:
            # Don't fail, just warn - processing might legitimately alter content
            print(f"  ⚠ Content similarity: {similarity:.0%} (threshold: {min_similarity:.0%})")
            
        return similarity >= min_similarity


class WorkflowState:
    """Manage workflow state in a more structured way."""
    
    def __init__(self):
        self.phases = {
            "setup": {"status": "pending", "data": {}},
            "upload": {"status": "pending", "data": []},
            "processing": {"status": "pending", "data": []},
            "interaction": {"status": "pending", "data": []},
            "cleanup": {"status": "pending", "data": []}
        }
        self.current_phase = "setup"
    
    def enter_phase(self, phase_name: str):
        """Mark entry into a new phase."""
        if self.current_phase:
            self.phases[self.current_phase]["status"] = "completed"
        self.current_phase = phase_name
        self.phases[phase_name]["status"] = "in_progress"
        print(f"\n{'='*60}")
        print(f"ENTERING PHASE: {phase_name.upper()}")
        print(f"{'='*60}")
    
    def add_phase_data(self, data: Any):
        """Add data to current phase."""
        phase = self.phases[self.current_phase]
        if isinstance(phase["data"], list):
            phase["data"].append(data)
        elif isinstance(phase["data"], dict):
            phase["data"].update(data)
    
    def get_phase_data(self, phase_name: str) -> Any:
        """Get data from a specific phase."""
        return self.phases.get(phase_name, {}).get("data")
    
    def verify_phase_complete(self, phase_name: str) -> bool:
        """Verify a phase completed successfully."""
        phase = self.phases.get(phase_name)
        if not phase:
            return False
        
        if phase["status"] != "completed":
            return False
        
        # Check phase has data
        data = phase["data"]
        if isinstance(data, list):
            return len(data) > 0
        elif isinstance(data, dict):
            return len(data) > 0
        
        return True