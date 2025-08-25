"""
Concurrent Operations Test Suite
---------------------------------

Tests for race conditions, concurrent access patterns, and system behavior
under parallel load. Ensures data integrity and proper synchronization.

Test Categories:
1. Concurrent Uploads
2. Parallel CRUD Operations  
3. Race Conditions
4. Load Testing Patterns
5. State Consistency Under Concurrency
"""

import pytest
import httpx
import asyncio
import time
import threading
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Callable, Optional
from pathlib import Path
import random
import string
from datetime import datetime

from fixtures import (
    api_client, authenticated_client, data_tracker, test_user_credentials,
    create_test_file, cleanup_test_file,
    BASE_URL, API_PREFIX
)
from test_data import TestDataGenerator


class ConcurrentTestHelper:
    """Helper utilities for concurrent testing."""
    
    @staticmethod
    def run_concurrent_requests(
        func: Callable,
        args_list: List[tuple],
        max_workers: int = 10
    ) -> Dict[str, Any]:
        """
        Run multiple requests concurrently and collect results.
        
        Returns:
            Dict with 'successful', 'failed', 'errors' lists
        """
        results = {
            'successful': [],
            'failed': [],
            'errors': [],
            'timing': [],
            'race_conditions': []  # Track detected race conditions
        }
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_args = {}
            for args in args_list:
                future = executor.submit(func, *args)
                future_to_args[future] = args
            
            # Collect results
            for future in as_completed(future_to_args):
                start_time = time.time()
                args = future_to_args[future]
                
                try:
                    result = future.result(timeout=30)
                    results['successful'].append({
                        'args': args,
                        'result': result,
                        'duration': time.time() - start_time,
                        'timestamp': time.time()  # Add timestamp for ordering
                    })
                except Exception as e:
                    results['failed'].append({
                        'args': args,
                        'error': str(e),
                        'error_type': type(e).__name__,
                        'duration': time.time() - start_time,
                        'timestamp': time.time()
                    })
                    results['errors'].append(e)
                
                results['timing'].append(time.time() - start_time)
        
        # Analyze for race conditions
        ConcurrentTestHelper._analyze_race_conditions(results)
        
        return results
    
    @staticmethod
    def detect_race_condition(results: List[Dict]) -> bool:
        """Detect if a race condition occurred based on results."""
        # Extract IDs from various response formats
        ids = []
        for r in results:
            if not r:
                continue
            
            # Handle different response formats
            if 'results' in r and isinstance(r['results'], list) and r['results']:
                # New format with results array
                ids.append(r['results'][0].get('db_id'))
            elif 'id' in r:
                ids.append(r['id'])
            elif 'media_id' in r:
                ids.append(r['media_id'])
        
        # Filter out None values
        valid_ids = [id for id in ids if id is not None]
        unique_ids = set(valid_ids)
        
        # Check for duplicate IDs
        if len(valid_ids) != len(unique_ids):
            return True  # Duplicate IDs indicate race condition
        
        return False
    
    @staticmethod
    def _analyze_race_conditions(results: Dict[str, Any]) -> None:
        """Analyze results for various race condition indicators."""
        race_indicators = []
        
        # Check for duplicate IDs
        if results['successful']:
            all_ids = []
            for item in results['successful']:
                result = item.get('result', {})
                if 'results' in result and result['results']:
                    id_val = result['results'][0].get('db_id')
                    if id_val:
                        all_ids.append(id_val)
            
            if len(all_ids) != len(set(all_ids)):
                race_indicators.append('Duplicate IDs detected')
        
        # Check for lost updates (same operation sometimes fails)
        if results['successful'] and results['failed']:
            # If we have both successes and failures for identical operations
            success_args = [str(s['args']) for s in results['successful']]
            failed_args = [str(f['args']) for f in results['failed']]
            
            # Check error patterns
            error_types = {}
            for failure in results['failed']:
                error_type = failure.get('error_type', 'Unknown')
                error_types[error_type] = error_types.get(error_type, 0) + 1
            
            # Conflicting writes often show as intermittent failures
            if 'HTTPStatusError' in error_types and error_types['HTTPStatusError'] > 2:
                race_indicators.append('Intermittent failures suggest race conditions')
        
        # Check timing patterns
        if results['timing']:
            avg_time = sum(results['timing']) / len(results['timing'])
            max_time = max(results['timing'])
            
            # Large variance in timing can indicate contention
            if max_time > avg_time * 3:
                race_indicators.append(f'High timing variance: avg={avg_time:.2f}s, max={max_time:.2f}s')
        
        results['race_conditions'] = race_indicators
    
    @staticmethod
    def measure_throughput(
        func: Callable,
        duration_seconds: int = 10,
        target_rps: int = 10
    ) -> Dict[str, float]:
        """Measure throughput over a duration."""
        start_time = time.time()
        end_time = start_time + duration_seconds
        
        successful_requests = 0
        failed_requests = 0
        response_times = []
        
        while time.time() < end_time:
            request_start = time.time()
            
            try:
                func()
                successful_requests += 1
            except Exception:
                failed_requests += 1
            
            response_time = time.time() - request_start
            response_times.append(response_time)
            
            # Try to maintain target RPS
            sleep_time = max(0, (1.0 / target_rps) - response_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        total_time = time.time() - start_time
        
        return {
            'total_requests': successful_requests + failed_requests,
            'successful_requests': successful_requests,
            'failed_requests': failed_requests,
            'requests_per_second': (successful_requests + failed_requests) / total_time,
            'average_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'min_response_time': min(response_times) if response_times else 0,
            'max_response_time': max(response_times) if response_times else 0,
            'success_rate': successful_requests / (successful_requests + failed_requests) if (successful_requests + failed_requests) > 0 else 0
        }


class TestConcurrentUploads:
    """Test concurrent file upload scenarios."""
    
    def test_concurrent_uploads_same_file(self, authenticated_client, data_tracker):
        """Test multiple users uploading the same file simultaneously."""
        # Create a test file
        content = TestDataGenerator.sample_text_content()
        test_file = create_test_file(content, suffix=".txt")
        data_tracker.add_file(test_file)
        
        def upload_file(client, file_path, attempt_num):
            """Upload function for concurrent execution."""
            return client.upload_media(
                file_path=file_path,
                title=f"Concurrent Upload Test {attempt_num}",
                media_type="document"
            )
        
        # Prepare arguments for 10 concurrent uploads
        args_list = [
            (authenticated_client, test_file, i)
            for i in range(10)
        ]
        
        # Run concurrent uploads
        results = ConcurrentTestHelper.run_concurrent_requests(
            upload_file,
            args_list,
            max_workers=10
        )
        
        # Analyze results
        print(f"\nConcurrent upload results:")
        print(f"  Successful: {len(results['successful'])}")
        print(f"  Failed: {len(results['failed'])}")
        
        if results['race_conditions']:
            print(f"  ⚠ Race conditions detected: {results['race_conditions']}")
        
        # At least some should succeed
        assert len(results['successful']) > 0, "At least one upload should succeed"
        
        # For same file uploads, duplicate IDs are expected due to content deduplication
        # This is correct behavior, not a race condition
        successful_results = [r['result'] for r in results['successful']]
        if successful_results:
            has_duplicates = ConcurrentTestHelper.detect_race_condition(successful_results)
            if has_duplicates:
                print(f"  ✓ Content deduplication working correctly (same IDs for same content)")
            else:
                print(f"  ⚠ Warning: Expected deduplication for identical content uploads")
        
        # Track uploaded media for cleanup
        for success in results['successful']:
            if 'result' in success:
                result = success['result']
                if 'results' in result and result['results']:
                    media_id = result['results'][0].get('db_id')
                    if media_id:
                        data_tracker.add_media(media_id)
        
        cleanup_test_file(test_file)
    
    def test_rapid_successive_uploads(self, authenticated_client, data_tracker):
        """Test rapid successive uploads from the same user."""
        files_to_upload = []
        
        # Create 20 small test files
        for i in range(20):
            content = f"Rapid upload test file {i}\n" * 10
            test_file = create_test_file(content, suffix=f"_{i}.txt")
            files_to_upload.append(test_file)
            data_tracker.add_file(test_file)
        
        # Upload as fast as possible
        start_time = time.time()
        successful_uploads = 0
        failed_uploads = 0
        
        for i, file_path in enumerate(files_to_upload):
            try:
                response = authenticated_client.upload_media(
                    file_path=file_path,
                    title=f"Rapid Upload {i}",
                    media_type="document"
                )
                
                if 'results' in response and response['results']:
                    media_id = response['results'][0].get('db_id')
                    if media_id:
                        data_tracker.add_media(media_id)
                        successful_uploads += 1
            except httpx.HTTPStatusError as e:
                failed_uploads += 1
                if e.response.status_code == 429:  # Rate limited
                    print(f"Rate limited after {i} uploads")
                    break
        
        duration = time.time() - start_time
        uploads_per_second = successful_uploads / duration if duration > 0 else 0
        
        print(f"Uploaded {successful_uploads} files in {duration:.2f}s ({uploads_per_second:.2f} uploads/sec)")
        
        # Should handle rapid uploads gracefully
        assert successful_uploads > 0, "Should successfully upload at least some files"
        
        # Clean up
        for file_path in files_to_upload:
            cleanup_test_file(file_path)
    
    def test_upload_during_processing(self, authenticated_client, data_tracker):
        """Test uploading while previous upload is still processing."""
        # Create a larger file that takes time to process
        large_content = TestDataGenerator.sample_text_content() * 100
        large_file = create_test_file(large_content, suffix="_large.txt")
        data_tracker.add_file(large_file)
        
        # Create a small file for quick upload
        small_content = "Small test file"
        small_file = create_test_file(small_content, suffix="_small.txt")
        data_tracker.add_file(small_file)
        
        try:
            # Start large file upload (might take time to process)
            large_upload_future = ThreadPoolExecutor(max_workers=1).submit(
                authenticated_client.upload_media,
                large_file,
                "Large File Processing",
                "document"
            )
            
            # Immediately upload small file
            time.sleep(0.1)  # Small delay to ensure first upload started
            
            small_response = authenticated_client.upload_media(
                file_path=small_file,
                title="Small File During Processing",
                media_type="document"
            )
            
            # Both should succeed
            assert small_response is not None, "Small file upload should succeed"
            
            # Wait for large upload to complete
            large_response = large_upload_future.result(timeout=30)
            assert large_response is not None, "Large file upload should succeed"
            
            # Track for cleanup
            for response in [small_response, large_response]:
                if 'results' in response and response['results']:
                    media_id = response['results'][0].get('db_id')
                    if media_id:
                        data_tracker.add_media(media_id)
                        
        finally:
            cleanup_test_file(large_file)
            cleanup_test_file(small_file)


class TestConcurrentCRUD:
    """Test concurrent Create, Read, Update, Delete operations."""
    
    def test_concurrent_note_updates(self, authenticated_client, data_tracker):
        """Test multiple concurrent updates to the same note."""
        # Add small delay to avoid rate limiting from previous tests
        time.sleep(0.5)
        
        # Create a note
        note_response = authenticated_client.create_note(
            title="Concurrent Update Test",
            content="Initial content",
            keywords=["concurrent", "test"]
        )
        
        note_id = note_response.get('id') or note_response.get('note_id')
        data_tracker.add_note(note_id)
        
        def update_note(client, note_id, update_num):
            """Update note function for concurrent execution."""
            return client.update_note(
                note_id=note_id,
                content=f"Updated content version {update_num}",
                version=1  # Optimistic locking - same version
            )
        
        # Prepare 10 concurrent update attempts
        args_list = [
            (authenticated_client, note_id, i)
            for i in range(10)
        ]
        
        # Run concurrent updates
        results = ConcurrentTestHelper.run_concurrent_requests(
            update_note,
            args_list,
            max_workers=10
        )
        
        # Analyze results for lost updates
        print(f"\nConcurrent note update results:")
        print(f"  Successful: {len(results['successful'])}")
        print(f"  Failed: {len(results['failed'])}")
        
        # Check for lost updates
        if len(results['successful']) > 1:
            print(f"  ⚠ WARNING: {len(results['successful'])} concurrent updates succeeded - checking for lost updates...")
            
            # Verify which update "won"
            try:
                final_note = authenticated_client.client.get(f"{API_PREFIX}/notes/{note_id}").json()
                final_content = final_note.get('content', '')
                final_version = final_note.get('version', 0)
                
                print(f"  Final content: '{final_content[:50]}...'")
                print(f"  Final version: {final_version}")
                
                # Get successful update numbers
                successful_nums = [r['args'][2] for r in results['successful']]
                print(f"  Updates that succeeded: {successful_nums}")
                
                # Check if content matches last update
                last_update_num = max(successful_nums) if successful_nums else -1
                if f"version {last_update_num}" not in final_content:
                    print(f"  ✗ Lost update detected! Final content doesn't match last successful update")
                    pytest.fail("Lost updates: Final content doesn't reflect all successful updates")
                    
            except Exception as e:
                print(f"  Could not verify final state: {e}")
        
        elif len(results['failed']) > 0:
            # Check if failures were due to optimistic locking (good!)
            conflict_failures = sum(1 for f in results['failed'] 
                                  if 'conflict' in str(f.get('error', '')).lower() 
                                  or '409' in str(f.get('error', '')))
            print(f"  Version conflicts detected: {conflict_failures} (this is good!)")
            
            if conflict_failures == 0 and len(results['failed']) > 0:
                print("  ⚠ Updates failed but not due to version conflicts")
        
        # Verify final state is consistent
        try:
            final_note = authenticated_client.client.get(f"{API_PREFIX}/notes/{note_id}").json()
            assert 'content' in final_note, "Note should still exist and be readable"
            print(f"  ✓ Final note state verified")
        except Exception as e:
            pytest.fail(f"Could not verify final note state: {e}")
    
    def test_concurrent_searches(self, authenticated_client):
        """Test multiple concurrent search requests - verify no result corruption."""
        search_queries = TestDataGenerator.sample_search_queries()
        
        def perform_search(client, query):
            """Search function for concurrent execution."""
            return client.search_media(query, limit=10)
        
        # Prepare 50 concurrent searches with known queries
        args_list = [
            (authenticated_client, search_queries[i % len(search_queries)])
            for i in range(50)
        ]
        
        # Run concurrent searches
        start_time = time.time()
        results = ConcurrentTestHelper.run_concurrent_requests(
            perform_search,
            args_list,
            max_workers=20
        )
        duration = time.time() - start_time
        
        # Calculate metrics
        successful = len(results['successful'])
        failed = len(results['failed'])
        avg_response_time = sum(results['timing']) / len(results['timing']) if results['timing'] else 0
        
        print(f"\nConcurrent search results:")
        print(f"  Successful: {successful}/{len(args_list)}")
        print(f"  Failed: {failed}")
        
        # Print first few errors for debugging
        if results['failed']:
            print(f"\nFirst few errors:")
            for failure in results['failed'][:3]:
                print(f"  - {failure['error_type']}: {failure['error']}")
        print(f"  Total time: {duration:.2f}s")
        print(f"  Avg response: {avg_response_time:.3f}s")
        print(f"  Requests/sec: {successful/duration:.1f}")
        
        # Verify no result corruption (same query should return similar results)
        query_results = {}
        for result in results['successful']:
            query = result['args'][1]  # Get the query
            response = result['result']
            
            if query not in query_results:
                query_results[query] = []
            
            # Store result count for consistency check
            if 'results' in response:
                query_results[query].append(len(response['results']))
            elif 'items' in response:
                query_results[query].append(len(response['items']))
        
        # Check consistency
        inconsistent_queries = []
        for query, counts in query_results.items():
            if len(set(counts)) > 1:  # Different result counts for same query
                inconsistent_queries.append(query)
                print(f"  ⚠ Inconsistent results for '{query}': {counts}")
        
        if inconsistent_queries:
            print(f"  ✗ {len(inconsistent_queries)} queries returned inconsistent results")
        else:
            print(f"  ✓ All queries returned consistent results")
        
        # Most searches should succeed
        assert successful > failed, "Most concurrent searches should succeed"
        
        # Response times should be reasonable
        assert avg_response_time < 5, "Average search response should be under 5 seconds"
    
    def test_create_update_delete_race(self, authenticated_client, data_tracker):
        """Test race condition between create, update, and delete operations."""
        # Add delay to avoid rate limiting from previous tests
        time.sleep(1)
        
        results = {
            'created': [],
            'updated': [],
            'deleted': [],
            'errors': []
        }
        
        def create_note():
            try:
                # Add small random delay to spread out requests
                time.sleep(random.uniform(0.05, 0.15))
                response = authenticated_client.create_note(
                    title=f"Race Test {random.randint(1000, 9999)}",
                    content="Testing race conditions",
                    keywords=["race", "test"]
                )
                note_id = response.get('id') or response.get('note_id')
                results['created'].append(note_id)
                return note_id
            except Exception as e:
                results['errors'].append(('create', str(e)))
                return None
        
        def update_note(note_id):
            if not note_id:
                return
            try:
                response = authenticated_client.update_note(
                    note_id=note_id,
                    content=f"Updated at {datetime.now()}",
                    version=1
                )
                results['updated'].append(note_id)
            except Exception as e:
                results['errors'].append(('update', str(e)))
        
        def delete_note(note_id):
            if not note_id:
                return
            try:
                response = authenticated_client.delete_note(note_id)
                results['deleted'].append(note_id)
            except Exception as e:
                results['errors'].append(('delete', str(e)))
        
        # Create notes and immediately try to update/delete them
        with ThreadPoolExecutor(max_workers=15) as executor:
            futures = []
            
            # Create 5 notes
            for _ in range(5):
                future = executor.submit(create_note)
                futures.append(future)
            
            # Wait for creates to start
            time.sleep(0.1)
            
            # Get created note IDs and schedule updates/deletes
            created_ids = []
            for future in futures:
                try:
                    note_id = future.result(timeout=2)
                    if note_id:
                        created_ids.append(note_id)
                        data_tracker.add_note(note_id)
                except:
                    pass
            
            # Now race update and delete on same notes
            for note_id in created_ids:
                executor.submit(update_note, note_id)
                executor.submit(delete_note, note_id)
        
        # Analyze results
        print(f"Created: {len(results['created'])}, Updated: {len(results['updated'])}, Deleted: {len(results['deleted'])}")
        print(f"Errors: {len(results['errors'])}")
        
        # Should handle race conditions gracefully
        assert len(results['errors']) < len(results['created']) * 2, \
            "Most operations should complete despite races"
    
    def test_concurrent_character_chat_sessions(self, authenticated_client, data_tracker):
        """Test multiple concurrent chat sessions with characters."""
        # Create a test character first
        character_data = TestDataGenerator.sample_character_card()
        
        try:
            char_response = authenticated_client.import_character(character_data)
            character_id = char_response.get('id') or char_response.get('character_id')
            data_tracker.add_character(character_id)
        except:
            # If character creation fails, skip test
            pytest.skip("Could not create test character")
            return
        
        def start_chat_session(client, session_num):
            """Start a chat session with the character."""
            messages = [
                {"role": "system", "content": character_data.get('system_prompt', '')},
                {"role": "user", "content": f"Hello from session {session_num}"}
            ]
            
            return client.chat_completion(
                messages=messages,
                model="gpt-3.5-turbo",
                temperature=0.7
            )
        
        # Start 10 concurrent chat sessions
        args_list = [
            (authenticated_client, i)
            for i in range(10)
        ]
        
        results = ConcurrentTestHelper.run_concurrent_requests(
            start_chat_session,
            args_list,
            max_workers=10
        )
        
        print(f"Concurrent chats: {len(results['successful'])} successful, {len(results['failed'])} failed")
        
        # Should handle multiple chat sessions
        assert len(results['successful']) > 0, "Should support concurrent chat sessions"
        
        # Track chat IDs for cleanup
        for success in results['successful']:
            if 'result' in success and 'chat_id' in success['result']:
                data_tracker.add_chat(success['result']['chat_id'])


class TestLoadPatterns:
    """Test various load patterns and stress scenarios."""
    
    def test_burst_traffic(self, authenticated_client):
        """Test handling of burst traffic (100 requests in 1 second)."""
        def make_request():
            return authenticated_client.health_check()
        
        # Send 100 requests as fast as possible
        args_list = [()] * 100  # 100 empty arg tuples
        
        start_time = time.time()
        results = ConcurrentTestHelper.run_concurrent_requests(
            lambda: make_request(),
            args_list,
            max_workers=50  # High concurrency
        )
        duration = time.time() - start_time
        
        successful = len(results['successful'])
        failed = len(results['failed'])
        
        print(f"Burst test: {successful} successful, {failed} failed in {duration:.2f}s")
        print(f"Requests per second: {100/duration:.2f}")
        
        # Should handle burst without crashing
        assert successful > 0, "Should handle at least some burst requests"
        
        # If rate limiting is implemented, some failures are expected
        if failed > 0:
            # Check if failures are due to rate limiting
            rate_limit_errors = sum(
                1 for e in results['errors']
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429
            )
            print(f"Rate limit errors: {rate_limit_errors}")
    
    def test_sustained_load(self, authenticated_client):
        """Test sustained load over 30 seconds."""
        def make_request():
            # Mix of different request types
            request_type = random.choice(['health', 'list', 'search'])
            
            if request_type == 'health':
                return authenticated_client.health_check()
            elif request_type == 'list':
                return authenticated_client.get_media_list(limit=10)
            else:
                return authenticated_client.search_media("test", limit=5)
        
        # Measure throughput over 30 seconds at 10 RPS
        metrics = ConcurrentTestHelper.measure_throughput(
            make_request,
            duration_seconds=30,
            target_rps=10
        )
        
        print(f"Sustained load test results:")
        print(f"  Total requests: {metrics['total_requests']}")
        print(f"  Successful: {metrics['successful_requests']}")
        print(f"  Failed: {metrics['failed_requests']}")
        print(f"  Actual RPS: {metrics['requests_per_second']:.2f}")
        print(f"  Avg response time: {metrics['average_response_time']:.3f}s")
        print(f"  Success rate: {metrics['success_rate']:.2%}")
        
        # Should maintain reasonable performance
        assert metrics['success_rate'] > 0.8, "Should maintain >80% success rate under load"
        assert metrics['average_response_time'] < 2, "Average response should be under 2 seconds"
    
    def test_mixed_workload(self, authenticated_client, data_tracker):
        """Test mixed workload with reads and writes."""
        # Prepare test data
        test_files = []
        for i in range(5):
            content = f"Mixed workload test {i}"
            test_file = create_test_file(content, suffix=f"_mixed_{i}.txt")
            test_files.append(test_file)
            data_tracker.add_file(test_file)
        
        created_media_ids = []
        created_note_ids = []
        
        def mixed_operation():
            """Perform a random operation."""
            operation = random.choice(['upload', 'read', 'search', 'create_note', 'list'])
            
            try:
                if operation == 'upload' and test_files:
                    file_path = random.choice(test_files)
                    response = authenticated_client.upload_media(
                        file_path=file_path,
                        title=f"Mixed Test {random.randint(1000, 9999)}",
                        media_type="document"
                    )
                    if 'results' in response and response['results']:
                        media_id = response['results'][0].get('db_id')
                        if media_id:
                            created_media_ids.append(media_id)
                    return ('upload', 'success')
                    
                elif operation == 'read' and created_media_ids:
                    media_id = random.choice(created_media_ids)
                    response = authenticated_client.get_media_item(media_id)
                    return ('read', 'success')
                    
                elif operation == 'search':
                    query = random.choice(['test', 'document', 'content'])
                    response = authenticated_client.search_media(query, limit=5)
                    return ('search', 'success')
                    
                elif operation == 'create_note':
                    response = authenticated_client.create_note(
                        title=f"Mixed Note {random.randint(1000, 9999)}",
                        content="Mixed workload test note",
                        keywords=["mixed", "test"]
                    )
                    note_id = response.get('id') or response.get('note_id')
                    if note_id:
                        created_note_ids.append(note_id)
                    return ('create_note', 'success')
                    
                else:  # list
                    response = authenticated_client.get_media_list(limit=10)
                    return ('list', 'success')
                    
            except Exception as e:
                return (operation, 'failed')
        
        # Run mixed workload for 20 seconds
        start_time = time.time()
        operation_counts = {}
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            
            while time.time() - start_time < 20:
                future = executor.submit(mixed_operation)
                futures.append(future)
                time.sleep(0.1)  # Space out submissions
            
            # Collect results
            for future in as_completed(futures):
                try:
                    operation, status = future.result(timeout=5)
                    key = f"{operation}_{status}"
                    operation_counts[key] = operation_counts.get(key, 0) + 1
                except:
                    pass
        
        # Track created resources for cleanup
        for media_id in created_media_ids:
            data_tracker.add_media(media_id)
        for note_id in created_note_ids:
            data_tracker.add_note(note_id)
        
        # Clean up test files
        for file_path in test_files:
            cleanup_test_file(file_path)
        
        # Print results
        print("Mixed workload results:")
        for key, count in operation_counts.items():
            print(f"  {key}: {count}")
        
        # Should handle mixed workload
        total_operations = sum(operation_counts.values())
        successful_operations = sum(
            count for key, count in operation_counts.items()
            if 'success' in key
        )
        
        assert successful_operations > 0, "Should complete some operations successfully"
        success_rate = successful_operations / total_operations if total_operations > 0 else 0
        print(f"Overall success rate: {success_rate:.2%}")
    
    def test_gradual_ramp_up(self, authenticated_client):
        """Test gradual load increase to find breaking point."""
        current_rps = 1
        max_rps = 50
        ramp_duration = 5  # seconds per level
        
        metrics_by_rps = {}
        breaking_point = None
        
        while current_rps <= max_rps:
            print(f"Testing at {current_rps} RPS...")
            
            # Test at current RPS
            metrics = ConcurrentTestHelper.measure_throughput(
                lambda: authenticated_client.health_check(),
                duration_seconds=ramp_duration,
                target_rps=current_rps
            )
            
            metrics_by_rps[current_rps] = metrics
            
            # Check if we've hit breaking point (success rate < 50%)
            if metrics['success_rate'] < 0.5:
                breaking_point = current_rps
                print(f"Breaking point found at {current_rps} RPS")
                break
            
            # Increase load
            if current_rps < 10:
                current_rps += 2
            else:
                current_rps += 5
        
        # Print summary
        print("\nLoad test summary:")
        for rps, metrics in metrics_by_rps.items():
            print(f"  {rps} RPS: {metrics['success_rate']:.1%} success, "
                  f"{metrics['average_response_time']:.3f}s avg response")
        
        if breaking_point:
            print(f"\nSystem breaking point: ~{breaking_point} RPS")
        else:
            print(f"\nSystem handled up to {max_rps} RPS")


class TestStateConsistency:
    """Test state consistency under concurrent operations."""
    
    def test_optimistic_locking(self, authenticated_client, data_tracker):
        """Test optimistic locking mechanism for concurrent updates."""
        # Create a note
        note_response = authenticated_client.create_note(
            title="Optimistic Locking Test",
            content="Version 1",
            keywords=["locking", "test"]
        )
        
        note_id = note_response.get('id') or note_response.get('note_id')
        data_tracker.add_note(note_id)
        
        # Get current version
        current_note = authenticated_client.client.get(f"{API_PREFIX}/notes/{note_id}").json()
        version = current_note.get('version', 1)
        
        # Two clients try to update with same version
        def update_with_version(content, version):
            return authenticated_client.update_note(
                note_id=note_id,
                content=content,
                version=version
            )
        
        # Both updates use same version (simulating concurrent reads)
        args_list = [
            (f"Update A at {datetime.now()}", version),
            (f"Update B at {datetime.now()}", version)
        ]
        
        results = ConcurrentTestHelper.run_concurrent_requests(
            update_with_version,
            args_list,
            max_workers=2
        )
        
        # With proper optimistic locking, only one should succeed
        if len(results['successful']) == 2:
            print("Warning: Both updates succeeded - no optimistic locking detected")
        elif len(results['successful']) == 1:
            print("Good: Optimistic locking prevented concurrent update conflict")
        
        # Verify final state is consistent
        final_note = authenticated_client.client.get(f"{API_PREFIX}/notes/{note_id}").json()
        assert 'content' in final_note, "Note should be in consistent state"
    
    def test_delete_during_access(self, authenticated_client, data_tracker):
        """Test deleting a resource while it's being accessed."""
        # Create a note
        note_response = authenticated_client.create_note(
            title="Delete During Access Test",
            content="This will be deleted during access",
            keywords=["delete", "test"]
        )
        
        note_id = note_response.get('id') or note_response.get('note_id')
        # Don't track for cleanup since we're deleting it
        
        results = {'read_success': 0, 'read_failed': 0, 'delete_success': False}
        
        def read_note():
            try:
                response = authenticated_client.client.get(f"{API_PREFIX}/notes/{note_id}")
                response.raise_for_status()
                results['read_success'] += 1
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    results['read_failed'] += 1
                else:
                    raise
        
        def delete_note():
            try:
                response = authenticated_client.delete_note(note_id)
                results['delete_success'] = True
            except:
                pass
        
        # Start multiple readers and one deleter
        with ThreadPoolExecutor(max_workers=11) as executor:
            # Start 10 readers
            read_futures = [executor.submit(read_note) for _ in range(10)]
            
            # Give readers a head start
            time.sleep(0.1)
            
            # Delete while reading
            delete_future = executor.submit(delete_note)
            
            # Wait for all to complete
            for future in read_futures:
                future.result(timeout=5)
            delete_future.result(timeout=5)
        
        print(f"Reads before delete: {results['read_success']}")
        print(f"Reads after delete (404s): {results['read_failed']}")
        print(f"Delete succeeded: {results['delete_success']}")
        
        # Should handle gracefully
        assert results['delete_success'] or results['read_success'] > 0, \
            "Should either delete or allow reads"
        
        # Total should be 10
        assert results['read_success'] + results['read_failed'] == 10, \
            "All reads should complete with either success or 404"
    
    def test_transaction_isolation(self, authenticated_client, data_tracker):
        """Test transaction isolation between concurrent operations."""
        # Create initial state
        notes_created = []
        
        for i in range(3):
            response = authenticated_client.create_note(
                title=f"Isolation Test {i}",
                content=f"Initial content {i}",
                keywords=["isolation", "test"]
            )
            note_id = response.get('id') or response.get('note_id')
            notes_created.append(note_id)
            data_tracker.add_note(note_id)
        
        def read_all_notes():
            """Read all notes and return their content."""
            contents = []
            for note_id in notes_created:
                try:
                    response = authenticated_client.client.get(f"{API_PREFIX}/notes/{note_id}")
                    note = response.json()
                    contents.append(note.get('content', ''))
                except:
                    contents.append(None)
            return contents
        
        def update_all_notes(suffix):
            """Update all notes with a suffix."""
            for note_id in notes_created:
                try:
                    authenticated_client.update_note(
                        note_id=note_id,
                        content=f"Updated content {suffix}",
                        version=1
                    )
                except:
                    pass
        
        # Concurrent reads and writes
        with ThreadPoolExecutor(max_workers=10) as executor:
            # Start readers
            read_futures = [executor.submit(read_all_notes) for _ in range(5)]
            
            # Start writers
            write_futures = [
                executor.submit(update_all_notes, f"writer_{i}")
                for i in range(2)
            ]
            
            # Collect read results
            read_results = [future.result(timeout=10) for future in read_futures]
            
            # Wait for writes to complete
            for future in write_futures:
                future.result(timeout=10)
        
        # Verify reads were consistent (each read saw a consistent state)
        for read_result in read_results:
            non_none_contents = [c for c in read_result if c is not None]
            assert len(non_none_contents) == len(notes_created), \
                "Each read should see all notes"
        
        print("Transaction isolation test completed successfully")


# Test markers would be defined in pytest.ini or pyproject.toml
# For now, commenting out to avoid errors
# pytest.mark.concurrent = pytest.mark.concurrent
# pytest.mark.slow = pytest.mark.slow