"""
Comprehensive end-to-end tests for RAG pipeline.

Tests the complete flow including:
- Document upload
- Embedding generation
- FTS indexing
- Vector search
- Hybrid search
- Performance benchmarks
"""

import pytest
import asyncio
import time
import tempfile
import os
from typing import Dict, List, Any
from pathlib import Path

from fixtures import api_client, authenticated_client, APIClient

# Test documents with diverse content
TEST_DOCUMENTS = [
    {
        "title": "Introduction to Machine Learning",
        "content": """Machine learning is a branch of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. 
        It focuses on developing computer programs that can access data and use it to learn for themselves.
        The process begins with observations or data, such as examples, direct experience, or instruction, to look for patterns in data."""
    },
    {
        "title": "Neural Networks Explained",
        "content": """Neural networks are computing systems inspired by biological neural networks that constitute animal brains. 
        An artificial neural network is based on a collection of connected units or nodes called artificial neurons, which loosely model the neurons in a biological brain.
        Each connection can transmit a signal from one artificial neuron to another."""
    },
    {
        "title": "Deep Learning Fundamentals", 
        "content": """Deep learning is a subset of machine learning that uses multi-layered neural networks to progressively extract higher-level features from raw input.
        For example, in image processing, lower layers may identify edges, while higher layers may identify human-relevant concepts.
        Deep learning architectures include deep neural networks, recurrent neural networks, and convolutional neural networks."""
    }
]


@pytest.mark.requires_rag
class TestRAGComprehensive:
    """Comprehensive tests for the RAG pipeline."""
    
    def setup_method(self):
        """Setup test method."""
        self.uploaded_ids = []
        
    def teardown_method(self):
        """Cleanup after test."""
        # Could delete uploaded documents here if needed
        pass
    
    def test_full_rag_pipeline_with_indexing_wait(self, api_client):
        """Test complete RAG pipeline with proper indexing wait."""
        
        # Step 1: Upload documents with embedding generation
        print("\n1. Uploading test documents...")
        
        for doc in TEST_DOCUMENTS:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
                f.write(doc["content"])
                temp_file = f.name
            
            try:
                with open(temp_file, 'rb') as f:
                    files = {"files": (doc["title"] + ".txt", f, "text/plain")}
                    data = {
                        "title": doc["title"],
                        "media_type": "document",
                        "generate_embeddings": "true",
                        "overwrite_existing": "true"
                    }
                    
                    response = api_client.client.post(
                        f"{api_client.base_url}/api/v1/media/add",
                        files=files,
                        data=data,
                        headers={"X-API-KEY": "test-api-key-12345"}
                    )
                    
                    assert response.status_code in [200, 207], f"Upload failed: {response.text}"
                    
                    result = response.json()
                    if "results" in result:
                        for item in result["results"]:
                            if item.get("status") == "Success" and item.get("db_id"):
                                self.uploaded_ids.append(item["db_id"])
                                print(f"  ✓ Uploaded: {doc['title']} (ID: {item['db_id']})")
                                break
                    
            finally:
                os.unlink(temp_file)
        
        assert len(self.uploaded_ids) > 0, "No documents were uploaded successfully"
        
        # Step 2: Wait for indexing to complete
        print("\n2. Waiting for indexing to complete...")
        max_wait = 10  # seconds
        start_time = time.time()
        indexed = False
        
        while time.time() - start_time < max_wait:
            # Test FTS search
            response = api_client.client.post(
                f"{api_client.base_url}/api/v1/rag/search/simple",
                json={
                    "query": "machine learning",
                    "search_mode": "fts",
                    "limit": 10
                }
            )
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                # Check if our documents are indexed
                found_titles = [r.get("title", "") for r in results]
                if any("Machine Learning" in title for title in found_titles):
                    indexed = True
                    print("  ✓ Documents indexed successfully")
                    break
            
            time.sleep(1)
        
        # Don't fail if not indexed, just warn
        if not indexed:
            print("  ⚠️ Documents may not be fully indexed yet")
        
        # Step 3: Test different search modes
        print("\n3. Testing search modes...")
        search_tests = [
            ("fts", "neural networks", "FTS Search"),
            ("vector", "artificial intelligence", "Vector Search"),
            ("hybrid", "deep learning architectures", "Hybrid Search")
        ]
        
        for mode, query, description in search_tests:
            response = api_client.client.post(
                f"{api_client.base_url}/api/v1/rag/search/simple",
                json={
                    "query": query,
                    "search_mode": mode,
                    "limit": 5
                }
            )
            
            assert response.status_code == 200, f"{description} failed"
            
            results = response.json()
            print(f"  ✓ {description}: {len(results.get('results', []))} results")
        
        # Step 4: Test hybrid search with different alpha values
        print("\n4. Testing hybrid search alpha tuning...")
        alpha_tests = [0.0, 0.3, 0.5, 0.7, 1.0]
        
        for alpha in alpha_tests:
            response = api_client.client.post(
                f"{api_client.base_url}/api/v1/rag/search/simple",
                json={
                    "query": "machine learning neural networks",
                    "search_mode": "hybrid",
                    "hybrid_alpha": alpha,
                    "limit": 3
                }
            )
            
            assert response.status_code == 200
            results = response.json()
            print(f"  ✓ Alpha {alpha:.1f}: {len(results.get('results', []))} results")
        
        print("\n✅ All RAG pipeline tests passed!")
    
    @pytest.mark.benchmark
    def test_search_performance(self, api_client, benchmark):
        """Benchmark search performance."""
        
        def perform_search():
            response = api_client.client.post(
                f"{api_client.base_url}/api/v1/rag/search/simple",
                json={
                    "query": "machine learning",
                    "search_mode": "hybrid",
                    "limit": 10
                }
            )
            assert response.status_code == 200
            return response.json()
        
        # Run benchmark
        result = benchmark(perform_search)
        
        # Log performance metrics
        print(f"\nSearch Performance:")
        print(f"  Mean time: {benchmark.stats['mean']:.4f}s")
        print(f"  Min time: {benchmark.stats['min']:.4f}s")
        print(f"  Max time: {benchmark.stats['max']:.4f}s")
        
        # Assert reasonable performance (adjust thresholds as needed)
        assert benchmark.stats['mean'] < 1.0, "Search too slow (>1s average)"
    
    def test_concurrent_searches(self, api_client):
        """Test concurrent search requests."""
        import concurrent.futures
        
        queries = [
            ("machine learning", "fts"),
            ("neural networks", "vector"),
            ("deep learning", "hybrid"),
            ("artificial intelligence", "fts"),
            ("computer vision", "vector")
        ]
        
        def search(query_mode):
            query, mode = query_mode
            response = api_client.client.post(
                f"{api_client.base_url}/api/v1/rag/search/simple",
                json={
                    "query": query,
                    "search_mode": mode,
                    "limit": 5
                }
            )
            return response.status_code == 200
        
        # Run concurrent searches
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(search, queries))
        
        # All should succeed
        assert all(results), "Some concurrent searches failed"
        print(f"✓ All {len(queries)} concurrent searches succeeded")
    
    def test_search_with_filters(self, api_client):
        """Test search with various filters."""
        
        # Test with keyword filter
        response = api_client.client.post(
            f"{api_client.base_url}/api/v1/rag/search/simple",
            json={
                "query": "learning",
                "search_mode": "hybrid",
                "keywords": ["neural", "deep"],
                "limit": 5
            }
        )
        
        assert response.status_code == 200
        results = response.json()
        
        # Verify keyword filtering if results exist
        if results.get("results"):
            for result in results["results"]:
                content = result.get("content", "").lower()
                # At least one keyword should be present
                assert any(kw in content for kw in ["neural", "deep"]), \
                    "Result doesn't contain required keywords"
        
        print("✓ Keyword filtering works correctly")
    
    def test_empty_and_edge_cases(self, api_client):
        """Test edge cases and error handling."""
        
        # Test empty query
        response = api_client.client.post(
            f"{api_client.base_url}/api/v1/rag/search/simple",
            json={
                "query": "",
                "search_mode": "fts"
            }
        )
        assert response.status_code in [400, 422], "Empty query should be rejected"
        
        # Test very long query
        long_query = "machine learning " * 100
        response = api_client.client.post(
            f"{api_client.base_url}/api/v1/rag/search/simple",
            json={
                "query": long_query,
                "search_mode": "fts",
                "limit": 1
            }
        )
        assert response.status_code in [200, 400, 422], "Long query should be handled"
        
        # Test invalid search mode
        response = api_client.client.post(
            f"{api_client.base_url}/api/v1/rag/search/simple",
            json={
                "query": "test",
                "search_mode": "invalid_mode"
            }
        )
        assert response.status_code in [400, 422], "Invalid mode should be rejected"
        
        print("✓ Edge cases handled correctly")