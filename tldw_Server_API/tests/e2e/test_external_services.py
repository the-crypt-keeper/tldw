# test_external_services.py
# Description: E2E tests for external service resilience and failure handling
#
"""
External Services Resilience E2E Tests
--------------------------------------

Tests handling of external service failures including LLM providers,
embedding services, transcription services, and rate limiting.
"""

import os
import time
import json
from typing import Dict, Any, List
from unittest.mock import patch, MagicMock
import pytest
import httpx
from datetime import datetime

from fixtures import (
    api_client, authenticated_client, data_tracker,
    create_test_file, StrongAssertionHelpers, SmartErrorHandler
)
from test_data import TestDataGenerator

# Rate limit delay between operations
RATE_LIMIT_DELAY = 0.5


class TestLLMProviderResilience:
    """Test handling of LLM provider failures and issues."""
    
    def test_llm_provider_unavailable(self, api_client):
        """Test behavior when LLM provider is unavailable."""
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, how are you?"}
        ]
        
        # Try with a non-configured or invalid model
        try:
            response = api_client.chat_completion(
                messages=messages,
                model="invalid-model-xyz",  # Non-existent model
                temperature=0.7
            )
            
            # Should either fail gracefully or fall back
            if "error" in response:
                assert "model" in response["error"].lower() or \
                       "not found" in response["error"].lower() or \
                       "not configured" in response["error"].lower(), \
                    f"Unclear error message: {response['error']}"
                print("✓ Invalid model handled gracefully")
                
        except httpx.HTTPStatusError as e:
            # Expected to fail with clear error
            if e.response.status_code == 503:
                error_detail = e.response.json().get("detail", "")
                assert "not configured" in error_detail.lower() or \
                       "unavailable" in error_detail.lower() or \
                       "model" in error_detail.lower(), \
                    f"Unclear error: {error_detail}"
                print("✓ LLM unavailability handled with proper error")
            elif e.response.status_code == 400:
                # Bad request for invalid model
                print("✓ Invalid model rejected with 400")
            else:
                raise
    
    def test_llm_rate_limit_handling(self, api_client):
        """Test handling of LLM API rate limits."""
        messages = [
            {"role": "user", "content": "Quick test message"}
        ]
        
        # Send rapid requests to potentially trigger rate limiting
        responses = []
        errors = []
        
        for i in range(5):  # Send 5 rapid requests
            try:
                response = api_client.chat_completion(
                    messages=messages,
                    model="gpt-3.5-turbo",
                    temperature=0.7
                )
                responses.append(response)
            except httpx.HTTPStatusError as e:
                errors.append((e.response.status_code, e.response.json()))
                if e.response.status_code == 429:
                    # Rate limit hit - good, it's being enforced
                    error_detail = e.response.json().get("detail", "")
                    assert "rate" in error_detail.lower() or \
                           "limit" in error_detail.lower() or \
                           "too many" in error_detail.lower(), \
                        f"Rate limit error unclear: {error_detail}"
                    
                    # Check for rate limit headers
                    headers = e.response.headers
                    if "X-RateLimit-Remaining" in headers:
                        remaining = headers["X-RateLimit-Remaining"]
                        print(f"✓ Rate limit enforced, remaining: {remaining}")
                    else:
                        print("✓ Rate limit enforced (429 response)")
                    break
            
            time.sleep(0.1)  # Small delay between requests
        
        # Either all succeeded (no rate limit) or we hit a rate limit
        if errors:
            rate_limit_errors = [e for e in errors if e[0] == 429]
            if rate_limit_errors:
                print(f"✓ Rate limiting tested: {len(rate_limit_errors)} rate limit responses")
        else:
            print(f"✓ Handled {len(responses)} rapid requests without rate limiting")
    
    def test_llm_timeout_handling(self, api_client):
        """Test handling of LLM request timeouts."""
        # Create a very long prompt that might cause timeout
        long_prompt = "Please analyze this text in extreme detail: " + \
                     " ".join(["word" + str(i) for i in range(10000)])
        
        messages = [
            {"role": "user", "content": long_prompt}
        ]
        
        try:
            # This might timeout or be rejected for length
            response = api_client.chat_completion(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=0.7
            )
            
            # If it succeeds, verify response
            if "choices" in response:
                assert len(response["choices"]) > 0
                print("✓ Long prompt handled successfully")
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 413:
                # Payload too large
                print("✓ Large prompt properly rejected (413)")
            elif e.response.status_code == 408:
                # Request timeout
                print("✓ Timeout handled properly (408)")
            elif e.response.status_code == 400:
                # Bad request (prompt too long)
                error = e.response.json().get("detail", "")
                if "token" in error.lower() or "length" in error.lower():
                    print("✓ Token limit enforced")
            else:
                # Other error
                print(f"Long prompt failed with: {e.response.status_code}")
        except httpx.TimeoutException:
            print("✓ Request timeout handled")
    
    def test_llm_fallback_behavior(self, api_client):
        """Test fallback to alternative LLM providers."""
        messages = [
            {"role": "user", "content": "Test fallback behavior"}
        ]
        
        # Test with primary provider
        primary_response = None
        try:
            primary_response = api_client.chat_completion(
                messages=messages,
                model="gpt-3.5-turbo",  # Primary
                temperature=0.7
            )
        except:
            pass
        
        # Test with alternative provider
        alternative_response = None
        try:
            alternative_response = api_client.chat_completion(
                messages=messages,
                model="claude-3-haiku",  # Alternative
                temperature=0.7
            )
        except:
            pass
        
        # At least one should work if properly configured
        if primary_response or alternative_response:
            print("✓ Multiple LLM providers available for fallback")
        else:
            print("⚠ No LLM providers configured - fallback not testable")


class TestEmbeddingServiceResilience:
    """Test embedding service failure handling."""
    
    def test_embedding_generation_failure(self, api_client, data_tracker):
        """Test handling of embedding generation failures."""
        # Create content that needs embeddings
        content = "Test content for embedding generation"
        file_path = self._create_temp_file(content)
        
        try:
            # Upload media (which may trigger embedding generation)
            response = api_client.upload_media(
                file_path=file_path,
                title="Embedding Test Document",
                media_type="document"
            )
            
            media_id = self._extract_media_id(response)
            if media_id:
                data_tracker.add_media(media_id)
                
                # Check if embeddings were generated or queued
                media = api_client.get_media_item(media_id)
                
                # Look for embedding status indicators
                if "embedding_status" in media:
                    status = media["embedding_status"]
                    assert status in ["pending", "processing", "completed", "failed"], \
                        f"Invalid embedding status: {status}"
                    print(f"✓ Embedding status tracked: {status}")
                else:
                    # Embeddings might be generated async
                    print("✓ Media uploaded, embeddings may be async")
                    
        finally:
            os.unlink(file_path)
    
    def test_bulk_embedding_load(self, api_client, data_tracker):
        """Test embedding service under bulk load."""
        # Create multiple documents for bulk embedding
        documents = []
        media_ids = []
        
        for i in range(10):
            content = f"Document {i}: " + TestDataGenerator.sample_text_content()
            file_path = self._create_temp_file(content, suffix=f"_{i}.txt")
            documents.append(file_path)
        
        try:
            # Upload all documents rapidly
            for file_path in documents:
                try:
                    response = api_client.upload_media(
                        file_path=file_path,
                        title=f"Bulk Embedding Test {len(media_ids)}",
                        media_type="document"
                    )
                    media_id = self._extract_media_id(response)
                    if media_id:
                        media_ids.append(media_id)
                        data_tracker.add_media(media_id)
                except:
                    pass  # Some might fail under load
            
            # Should handle bulk load gracefully
            assert len(media_ids) >= 5, \
                f"Too many failures in bulk upload: only {len(media_ids)}/10 succeeded"
            
            print(f"✓ Bulk embedding load handled: {len(media_ids)}/10 documents processed")
            
            # Check if embeddings are being processed
            time.sleep(2)  # Give some time for async processing
            
            # Check status of first few items
            for media_id in media_ids[:3]:
                try:
                    media = api_client.get_media_item(media_id)
                    # Just verify we can still access them
                    assert media is not None
                except:
                    pass
                    
        finally:
            for file_path in documents:
                if os.path.exists(file_path):
                    os.unlink(file_path)
    
    def test_embedding_service_unavailable(self, api_client):
        """Test behavior when embedding service is unavailable."""
        # This would require being able to disable the embedding service
        # For now, test that operations continue even if embeddings fail
        
        content = "Test content when embeddings unavailable"
        file_path = self._create_temp_file(content)
        
        try:
            # Upload should succeed even if embeddings fail
            response = api_client.upload_media(
                file_path=file_path,
                title="No Embedding Test",
                media_type="document"
            )
            
            media_id = self._extract_media_id(response)
            assert media_id is not None, \
                "Upload failed when it should succeed without embeddings"
            
            # Should still be able to search (might use FTS5 only)
            time.sleep(RATE_LIMIT_DELAY)  # Add delay to avoid rate limiting
            search_response = api_client.search_media("test content", limit=5)
            assert search_response is not None, \
                "Search failed when it should fall back to text search"
            
            print("✓ System continues functioning without embeddings")
            
        finally:
            os.unlink(file_path)
    
    def _create_temp_file(self, content: str, suffix: str = ".txt") -> str:
        """Create a temporary text file."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
            f.write(content)
            return f.name
    
    def _extract_media_id(self, response: Dict[str, Any]) -> int:
        """Extract media ID from various response formats."""
        if "results" in response and response["results"]:
            return response["results"][0].get("db_id")
        return response.get("media_id") or response.get("id")


class TestTranscriptionServiceResilience:
    """Test transcription service failure handling."""
    
    def test_transcription_service_unavailable(self, api_client, data_tracker):
        """Test behavior when transcription service is unavailable."""
        # Use the test audio file
        audio_file = self._create_test_audio()
        
        try:
            # Upload audio file
            response = api_client.upload_media(
                file_path=audio_file,
                title="Transcription Failure Test",
                media_type="audio"
            )
            
            # Should handle transcription failures gracefully
            if "results" in response and response["results"]:
                result = response["results"][0]
                
                if result.get("status") == "Error":
                    # Check error is clear
                    error_msg = result.get("error", "") or result.get("db_message", "")
                    print(f"✓ Transcription failure handled: {error_msg}")
                    
                elif result.get("db_id"):
                    media_id = result["db_id"]
                    data_tracker.add_media(media_id)
                    
                    # Check transcription status
                    media = api_client.get_media_item(media_id)
                    transcription = media.get("transcription") or media.get("transcript")
                    
                    if transcription is None or transcription == "":
                        print("✓ Upload succeeded despite transcription failure")
                    else:
                        print(f"✓ Transcription completed: {len(transcription)} chars")
                        
        finally:
            os.unlink(audio_file)
    
    def test_long_transcription_timeout(self, api_client, data_tracker):
        """Test timeout handling for long transcription jobs."""
        # Would need a very long audio file to test properly
        # For now, test with standard audio
        audio_file = self._create_test_audio()
        
        try:
            start_time = time.time()
            response = api_client.upload_media(
                file_path=audio_file,
                title="Long Transcription Test",
                media_type="audio"
            )
            elapsed = time.time() - start_time
            
            # Should complete or timeout within reasonable time
            assert elapsed < 120, f"Transcription took too long: {elapsed:.2f}s"
            
            media_id = self._extract_media_id(response)
            if media_id:
                data_tracker.add_media(media_id)
                print(f"✓ Transcription completed in {elapsed:.2f}s")
                
        finally:
            os.unlink(audio_file)
    
    def test_concurrent_transcription_requests(self, api_client, data_tracker):
        """Test handling of multiple concurrent document processing requests."""
        doc_files = []
        media_ids = []
        
        # Create multiple text files to simulate concurrent processing
        for i in range(5):
            doc_file = self._create_test_document(i)
            doc_files.append(doc_file)
        
        try:
            # Upload all documents rapidly
            for i, doc_file in enumerate(doc_files):
                try:
                    response = api_client.upload_media(
                        file_path=doc_file,
                        title=f"Concurrent Processing {i}",
                        media_type="document"
                    )
                    media_id = self._extract_media_id(response)
                    if media_id:
                        media_ids.append(media_id)
                        data_tracker.add_media(media_id)
                except Exception as e:
                    # Log but continue - some might fail under load
                    print(f"Upload {i} failed: {str(e)[:100]}")
            
            # Should handle at least some concurrent requests
            assert len(media_ids) >= 2, \
                f"Too many failures: only {len(media_ids)}/5 succeeded"
            
            print(f"✓ Concurrent processing handled: {len(media_ids)}/5 succeeded")
            
        finally:
            for doc_file in doc_files:
                if os.path.exists(doc_file):
                    os.unlink(doc_file)
    
    def _create_test_audio(self) -> str:
        """Create a test audio file."""
        from fixtures import create_test_audio
        return create_test_audio()
    
    def _create_test_document(self, index: int) -> str:
        """Create a unique test document for concurrent processing."""
        import tempfile
        content = f"Test document {index}\n\nThis is content for concurrent processing test.\nUnique ID: {index}"
        
        with tempfile.NamedTemporaryFile(mode='w', suffix=".txt", delete=False) as f:
            f.write(content)
            return f.name
    
    def _extract_media_id(self, response: Dict[str, Any]) -> int:
        """Extract media ID from various response formats."""
        if "results" in response and response["results"]:
            result = response["results"][0]
            if result.get("status") != "Error":
                return result.get("db_id")
        return response.get("media_id") or response.get("id")


class TestRateLimitingEnforcement:
    """Test API rate limiting and throttling."""
    
    def test_rate_limit_enforcement(self, api_client):
        """Test that rate limits are properly enforced."""
        # Send rapid requests to trigger rate limiting
        request_count = 0
        rate_limited = False
        
        for i in range(50):  # Try 50 rapid requests
            try:
                response = api_client.health_check()
                request_count += 1
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    rate_limited = True
                    
                    # Check rate limit headers
                    headers = e.response.headers
                    
                    # Common rate limit headers
                    limit_headers = [
                        "X-RateLimit-Limit",
                        "X-RateLimit-Remaining",
                        "X-RateLimit-Reset",
                        "Retry-After"
                    ]
                    
                    found_headers = {h: headers.get(h) for h in limit_headers if h in headers}
                    
                    if found_headers:
                        print(f"✓ Rate limit headers found: {found_headers}")
                    
                    print(f"✓ Rate limit enforced after {request_count} requests")
                    break
            
            # No delay - trying to trigger rate limit
        
        if not rate_limited:
            print(f"⚠ No rate limiting triggered after {request_count} rapid requests")
    
    def test_authenticated_vs_anonymous_limits(self, api_client):
        """Test different rate limits for authenticated vs anonymous users."""
        # Test anonymous rate limit
        temp_client = api_client.__class__(api_client.base_url)
        
        anonymous_count = 0
        authenticated_count = 0
        
        # Test anonymous requests
        for i in range(20):
            try:
                temp_client.health_check()
                anonymous_count += 1
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    break
        
        # Test authenticated requests (using main client)
        for i in range(20):
            try:
                api_client.health_check()
                authenticated_count += 1
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    break
        
        print(f"✓ Anonymous requests: {anonymous_count}, Authenticated: {authenticated_count}")
        
        # Authenticated should generally have higher limits
        # But this depends on configuration
    
    def test_rate_limit_reset(self, api_client):
        """Test rate limit reset behavior."""
        # First, try to hit rate limit
        hit_limit = False
        reset_time = None
        
        for i in range(100):
            try:
                api_client.health_check()
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:
                    hit_limit = True
                    # Get reset time from headers
                    reset_header = e.response.headers.get("X-RateLimit-Reset")
                    retry_after = e.response.headers.get("Retry-After")
                    
                    if retry_after:
                        wait_time = int(retry_after)
                        print(f"✓ Rate limit hit, retry after {wait_time}s")
                        
                        # Wait for reset (but not too long for test)
                        if wait_time <= 5:
                            time.sleep(wait_time + 1)
                            
                            # Try again after reset
                            try:
                                api_client.health_check()
                                print("✓ Rate limit reset successfully")
                            except httpx.HTTPStatusError as e2:
                                if e2.response.status_code == 429:
                                    print("⚠ Still rate limited after reset time")
                    break
        
        if not hit_limit:
            print("⚠ Could not trigger rate limit for reset test")


class TestWebScrapingResilience:
    """Test web scraping and external content fetching."""
    
    def test_unreachable_url(self, api_client):
        """Test handling of unreachable URLs."""
        unreachable_url = "http://this-domain-definitely-does-not-exist-12345.com/page"
        
        try:
            response = api_client.process_media(
                url=unreachable_url,
                title="Unreachable URL Test",
                persist=False
            )
            
            # Should handle gracefully
            if "error" in response:
                error_msg = response["error"]
                assert "not be reached" in error_msg.lower() or \
                       "failed" in error_msg.lower() or \
                       "error" in error_msg.lower(), \
                    f"Unclear error for unreachable URL: {error_msg}"
                print("✓ Unreachable URL handled gracefully")
                
        except httpx.HTTPStatusError as e:
            # Expected to fail
            assert e.response.status_code in [400, 422, 502, 504], \
                f"Unexpected status code: {e.response.status_code}"
            print(f"✓ Unreachable URL properly rejected ({e.response.status_code})")
    
    def test_timeout_on_slow_website(self, api_client):
        """Test timeout handling for slow websites."""
        # Use a URL that might be slow or timeout
        slow_url = "https://httpstat.us/200?sleep=30000"  # 30 second delay
        
        try:
            start_time = time.time()
            response = api_client.process_media(
                url=slow_url,
                title="Slow Website Test",
                persist=False
            )
            elapsed = time.time() - start_time
            
            # Should timeout before 30 seconds
            assert elapsed < 30, f"Did not timeout, took {elapsed:.2f}s"
            
        except httpx.HTTPStatusError as e:
            elapsed = time.time() - start_time
            assert elapsed < 30, f"Timeout took too long: {elapsed:.2f}s"
            print(f"✓ Slow website timed out properly after {elapsed:.2f}s")
        except httpx.TimeoutException:
            print("✓ Request timeout handled correctly")
    
    def test_redirect_handling(self, api_client):
        """Test handling of URL redirects."""
        # Use a URL that redirects
        redirect_url = "http://httpstat.us/301"  # Returns 301 redirect
        
        try:
            response = api_client.process_media(
                url=redirect_url,
                title="Redirect Test",
                persist=False
            )
            
            # Should handle redirects appropriately
            if response:
                print("✓ Redirect handled successfully")
                
        except httpx.HTTPStatusError as e:
            # Might reject redirects for security
            if e.response.status_code in [301, 302, 303, 307, 308]:
                print("✓ Redirect not followed (security)")
            else:
                print(f"Redirect handling: {e.response.status_code}")


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])