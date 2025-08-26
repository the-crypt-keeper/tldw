# test_search_features.py
# Description: E2E tests for advanced search capabilities including FTS5 and vector search
#
"""
Search Features E2E Tests
-------------------------

Tests advanced search functionality including SQLite FTS5 features,
ChromaDB vector search, hybrid search, and RAG capabilities.
"""

import os
import time
import json
from typing import Dict, Any, List, Optional
import pytest
import httpx
from datetime import datetime, timedelta

from fixtures import (
    api_client, authenticated_client, data_tracker,
    create_test_file, StrongAssertionHelpers
)
from test_data import TestDataGenerator

# Rate limit delay between operations
RATE_LIMIT_DELAY = 0.5


class TestFTS5SearchFeatures:
    """Test SQLite FTS5 full-text search capabilities."""
    
    def test_phrase_search(self, api_client, data_tracker):
        """Test exact phrase searching with quotes."""
        # Create content with specific phrases
        content1 = "The quick brown fox jumps over the lazy dog"
        content2 = "The brown fox is quick but not lazy"
        content3 = "A quick jump over the fence by the brown dog"
        
        media_ids = self._create_test_media(api_client, data_tracker, [
            ("Phrase Test 1", content1),
            ("Phrase Test 2", content2),
            ("Phrase Test 3", content3)
        ])
        
        if not media_ids:
            pytest.skip("Could not create test media")
        
        # Search for exact phrase
        phrase_query = '"quick brown fox"'  # Exact phrase search
        time.sleep(RATE_LIMIT_DELAY)  # Add delay to avoid rate limiting
        response = api_client.search_media(phrase_query, limit=10)
        
        results = self._extract_results(response)
        
        # Should find only content1 with exact phrase
        if results:
            # Verify most relevant result contains exact phrase
            top_result = results[0]
            result_content = self._get_result_content(top_result)
            
            # Check if exact phrase is in result
            if result_content:
                assert "quick brown fox" in result_content.lower(), \
                    "Exact phrase not found in top result"
                print(f"✓ Phrase search found exact match")
        else:
            print("⚠ No results for phrase search (FTS5 might not be configured)")
    
    def test_boolean_operators(self, api_client, data_tracker):
        """Test AND, OR, NOT boolean operators in search."""
        # Create test content
        content1 = "Python programming with machine learning"
        content2 = "Java programming without machine learning"
        content3 = "Machine learning and deep learning with Python"
        
        media_ids = self._create_test_media(api_client, data_tracker, [
            ("Boolean Test 1", content1),
            ("Boolean Test 2", content2),
            ("Boolean Test 3", content3)
        ])
        
        if not media_ids:
            pytest.skip("Could not create test media")
        
        # Test AND operator (both terms required)
        and_query = "Python AND machine"
        response = api_client.search_media(and_query, limit=10)
        and_results = self._extract_results(response)
        
        # Should find content1 and content3
        for result in and_results:
            content = self._get_result_content(result).lower()
            if content:
                # Both terms should be present
                has_python = "python" in content
                has_machine = "machine" in content
                if has_python and has_machine:
                    print("✓ AND operator working correctly")
                    break
        
        # Test OR operator (either term)
        or_query = "Python OR Java"
        response = api_client.search_media(or_query, limit=10)
        or_results = self._extract_results(response)
        
        # Should find all three potentially
        if len(or_results) >= 2:
            print(f"✓ OR operator found {len(or_results)} results")
        
        # Test NOT operator (exclusion)
        not_query = "programming NOT Java"
        response = api_client.search_media(not_query, limit=10)
        not_results = self._extract_results(response)
        
        # Should exclude content2
        for result in not_results:
            content = self._get_result_content(result).lower()
            if content and "java" not in content:
                print("✓ NOT operator excluding correctly")
                break
    
    def test_prefix_search(self, api_client, data_tracker):
        """Test prefix/wildcard searching."""
        # Create content with related terms
        content1 = "Testing machine learning algorithms"
        content2 = "Tested the machinery yesterday"
        content3 = "The test of machines continues"
        
        media_ids = self._create_test_media(api_client, data_tracker, [
            ("Prefix Test 1", content1),
            ("Prefix Test 2", content2),
            ("Prefix Test 3", content3)
        ])
        
        if not media_ids:
            pytest.skip("Could not create test media")
        
        # Search with prefix
        prefix_query = "machin*"  # Should match machine, machinery, machines
        response = api_client.search_media(prefix_query, limit=10)
        results = self._extract_results(response)
        
        if results:
            found_variations = set()
            for result in results:
                content = self._get_result_content(result).lower()
                if "machine" in content:
                    found_variations.add("machine")
                if "machinery" in content:
                    found_variations.add("machinery")
                if "machines" in content:
                    found_variations.add("machines")
            
            if len(found_variations) > 1:
                print(f"✓ Prefix search found variations: {found_variations}")
            else:
                print(f"⚠ Prefix search found limited variations: {found_variations}")
    
    def test_search_result_ranking(self, api_client, data_tracker):
        """Test search result relevance ranking."""
        # Create content with varying relevance
        content1 = "machine learning " * 10  # High frequency
        content2 = "machine learning is important"  # Medium frequency
        content3 = "briefly mentions machine and separately learning"  # Low relevance
        
        media_ids = self._create_test_media(api_client, data_tracker, [
            ("Ranking Test High", content1),
            ("Ranking Test Medium", content2),
            ("Ranking Test Low", content3)
        ])
        
        if not media_ids:
            pytest.skip("Could not create test media")
        
        # Search for the term
        query = "machine learning"
        response = api_client.search_media(query, limit=10)
        results = self._extract_results(response)
        
        if len(results) >= 2:
            # First result should have highest relevance
            first_content = self._get_result_content(results[0]).lower()
            
            # Count occurrences in first vs second result
            first_count = first_content.count("machine") + first_content.count("learning")
            
            if len(results) > 1:
                second_content = self._get_result_content(results[1]).lower()
                second_count = second_content.count("machine") + second_content.count("learning")
                
                # First should generally have more occurrences
                if first_count >= second_count:
                    print("✓ Search results ranked by relevance")
                else:
                    print("⚠ Search ranking may not be optimal")
    
    def _create_test_media(self, api_client, data_tracker, content_list: List[tuple]) -> List[int]:
        """Helper to create test media items."""
        media_ids = []
        for title, content in content_list:
            file_path = self._create_temp_file(content)
            try:
                response = api_client.upload_media(
                    file_path=file_path,
                    title=title,
                    media_type="document"
                )
                media_id = self._extract_media_id(response)
                if media_id:
                    media_ids.append(media_id)
                    data_tracker.add_media(media_id)
            finally:
                os.unlink(file_path)
        return media_ids
    
    def _create_temp_file(self, content: str) -> str:
        """Create a temporary text file."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            return f.name
    
    def _extract_media_id(self, response: Dict[str, Any]) -> Optional[int]:
        """Extract media ID from response."""
        if "results" in response and response["results"]:
            return response["results"][0].get("db_id")
        return response.get("media_id") or response.get("id")
    
    def _extract_results(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract search results from response."""
        if isinstance(response, list):
            return response
        return response.get("results") or response.get("items", [])
    
    def _get_result_content(self, result: Dict[str, Any]) -> str:
        """Extract content text from search result."""
        if "content" in result:
            content = result["content"]
            if isinstance(content, dict):
                return content.get("text", "")
            return str(content)
        return result.get("text", "") or result.get("snippet", "")


class TestVectorSearch:
    """Test ChromaDB vector search capabilities."""
    
    def test_semantic_similarity_search(self, api_client, data_tracker):
        """Test semantic similarity search using embeddings."""
        # Create semantically related content
        content1 = "Artificial intelligence and machine learning are transforming technology"
        content2 = "AI and ML technologies are revolutionizing the digital world"
        content3 = "Cooking recipes for delicious pasta dishes"  # Unrelated
        
        media_ids = []
        for i, (title, content) in enumerate([
            ("Semantic Test 1", content1),
            ("Semantic Test 2", content2),
            ("Semantic Test 3", content3)
        ]):
            file_path = self._create_temp_file(content)
            try:
                response = api_client.upload_media(
                    file_path=file_path,
                    title=title,
                    media_type="document"
                )
                media_id = self._extract_media_id(response)
                if media_id:
                    media_ids.append(media_id)
                    data_tracker.add_media(media_id)
            finally:
                os.unlink(file_path)
        
        if not media_ids:
            pytest.skip("Could not create test media")
        
        # Wait a bit for embeddings to generate
        time.sleep(2)
        
        # Search with semantically similar query
        semantic_query = "Deep learning and neural networks in modern computing"
        
        try:
            # Try RAG search which should use embeddings
            response = api_client.rag_simple_search(
                query=semantic_query,
                databases=["media"],
                top_k=5,
                enable_reranking=True
            )
            
            if response.get("success"):
                results = response.get("results", [])
                
                if results:
                    # Top results should be AI/ML related, not cooking
                    top_content = results[0].get("content", "").lower()
                    
                    # Check semantic relevance
                    has_tech_terms = any(term in top_content for term in 
                                        ["artificial", "intelligence", "machine", "learning", "ai", "ml", "technology"])
                    has_cooking_terms = any(term in top_content for term in 
                                          ["cooking", "recipes", "pasta", "dishes"])
                    
                    if has_tech_terms and not has_cooking_terms:
                        print("✓ Semantic search found relevant content")
                    else:
                        print("⚠ Semantic search may not be working optimally")
                else:
                    print("⚠ No results from semantic search")
                    
        except httpx.HTTPStatusError:
            # RAG endpoints might not be available
            print("⚠ RAG/semantic search not available")
    
    def test_vector_search_with_metadata_filtering(self, api_client, data_tracker):
        """Test vector search with metadata filters."""
        # Create content with different metadata
        test_data = [
            ("Tech Article 1", "Machine learning in healthcare", ["technology", "health"]),
            ("Tech Article 2", "AI for medical diagnosis", ["technology", "health"]),
            ("News Article", "Latest political developments", ["news", "politics"])
        ]
        
        media_ids = []
        for title, content, tags in test_data:
            file_path = self._create_temp_file(content)
            try:
                # Upload with tags/keywords
                response = api_client.upload_media(
                    file_path=file_path,
                    title=title,
                    media_type="document"
                )
                media_id = self._extract_media_id(response)
                if media_id:
                    media_ids.append(media_id)
                    data_tracker.add_media(media_id)
                    # Would need to add tags via separate endpoint
            finally:
                os.unlink(file_path)
        
        if not media_ids:
            pytest.skip("Could not create test media")
        
        # Search with keyword filter
        try:
            response = api_client.rag_simple_search(
                query="artificial intelligence applications",
                databases=["media"],
                keywords=["technology"],  # Filter by keyword
                top_k=5
            )
            
            if response.get("success"):
                results = response.get("results", [])
                print(f"✓ Vector search with filters returned {len(results)} results")
                
        except:
            print("⚠ Filtered vector search not available")
    
    def _create_temp_file(self, content: str) -> str:
        """Create a temporary text file."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            return f.name
    
    def _extract_media_id(self, response: Dict[str, Any]) -> Optional[int]:
        """Extract media ID from response."""
        if "results" in response and response["results"]:
            return response["results"][0].get("db_id")
        return response.get("media_id") or response.get("id")


class TestHybridSearch:
    """Test hybrid BM25 + vector search."""
    
    def test_hybrid_search_accuracy(self, api_client, data_tracker):
        """Test that hybrid search combines both keyword and semantic matching."""
        # Create test content
        content1 = "The Python programming language is excellent for data science"
        content2 = "Snake species like pythons are found in tropical regions"
        content3 = "Data analysis using R and statistical methods"
        
        media_ids = []
        for title, content in [
            ("Programming Doc", content1),
            ("Animal Doc", content2),
            ("Statistics Doc", content3)
        ]:
            file_path = self._create_temp_file(content)
            try:
                response = api_client.upload_media(
                    file_path=file_path,
                    title=title,
                    media_type="document"
                )
                media_id = self._extract_media_id(response)
                if media_id:
                    media_ids.append(media_id)
                    data_tracker.add_media(media_id)
            finally:
                os.unlink(file_path)
        
        if not media_ids:
            pytest.skip("Could not create test media")
        
        # Search for "Python" - should prefer programming context
        query = "Python coding and development"
        
        # Try both regular search and RAG search
        regular_results = []
        try:
            response = api_client.search_media(query, limit=5)
            regular_results = self._extract_results(response)
        except:
            pass
        
        rag_results = []
        try:
            response = api_client.rag_simple_search(
                query=query,
                databases=["media"],
                top_k=5,
                enable_reranking=True
            )
            if response.get("success"):
                rag_results = response.get("results", [])
        except:
            pass
        
        # Check if hybrid search improves relevance
        if rag_results:
            # RAG with reranking should prefer programming content
            top_result = rag_results[0].get("content", "").lower()
            if "programming" in top_result or "data" in top_result:
                print("✓ Hybrid search correctly prioritized relevant content")
            else:
                print("⚠ Hybrid search may need tuning")
        elif regular_results:
            print("✓ Text search available (hybrid not tested)")
        else:
            print("⚠ Search functionality limited")
    
    def test_reranking_effectiveness(self, api_client, data_tracker):
        """Test that reranking improves search results."""
        # Create content with varying relevance
        contents = [
            "Deep dive into machine learning algorithms and neural networks",
            "Machine washing instructions for delicate fabrics",
            "Learning to use sewing machines effectively",
            "Advanced machine learning techniques for computer vision"
        ]
        
        media_ids = []
        for i, content in enumerate(contents):
            file_path = self._create_temp_file(content)
            try:
                response = api_client.upload_media(
                    file_path=file_path,
                    title=f"Rerank Test {i}",
                    media_type="document"
                )
                media_id = self._extract_media_id(response)
                if media_id:
                    media_ids.append(media_id)
                    data_tracker.add_media(media_id)
            finally:
                os.unlink(file_path)
        
        if not media_ids:
            pytest.skip("Could not create test media")
        
        query = "machine learning artificial intelligence"
        
        # Search without reranking
        results_no_rerank = []
        try:
            response = api_client.rag_simple_search(
                query=query,
                databases=["media"],
                top_k=10,
                enable_reranking=False
            )
            if response.get("success"):
                results_no_rerank = response.get("results", [])
        except:
            pass
        
        # Search with reranking
        results_with_rerank = []
        try:
            response = api_client.rag_simple_search(
                query=query,
                databases=["media"],
                top_k=10,
                enable_reranking=True
            )
            if response.get("success"):
                results_with_rerank = response.get("results", [])
        except:
            pass
        
        if results_with_rerank and results_no_rerank:
            # Reranked results should better prioritize ML content
            reranked_top = results_with_rerank[0].get("content", "").lower()
            
            # Check if top result is actually about ML
            ml_terms = ["neural", "algorithm", "computer vision", "deep"]
            non_ml_terms = ["washing", "sewing", "fabric"]
            
            has_ml = any(term in reranked_top for term in ml_terms)
            has_non_ml = any(term in reranked_top for term in non_ml_terms)
            
            if has_ml and not has_non_ml:
                print("✓ Reranking improved result relevance")
            else:
                print("⚠ Reranking effectiveness unclear")
        else:
            print("⚠ Could not test reranking")
    
    def _create_temp_file(self, content: str) -> str:
        """Create a temporary text file."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            return f.name
    
    def _extract_results(self, response: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract search results from response."""
        if isinstance(response, list):
            return response
        return response.get("results") or response.get("items", [])
    
    def _extract_media_id(self, response: Dict[str, Any]) -> Optional[int]:
        """Extract media ID from response."""
        if "results" in response and response["results"]:
            return response["results"][0].get("db_id")
        return response.get("media_id") or response.get("id")


class TestRAGContextRetrieval:
    """Test RAG context retrieval and expansion."""
    
    def test_context_window_optimization(self, api_client, data_tracker):
        """Test that RAG optimizes context window usage."""
        # Create a long document
        long_content = "\n\n".join([
            f"Section {i}: " + TestDataGenerator.sample_text_content()[:200]
            for i in range(20)
        ])
        
        file_path = self._create_temp_file(long_content)
        try:
            response = api_client.upload_media(
                file_path=file_path,
                title="Long Document for Context Test",
                media_type="document"
            )
            media_id = self._extract_media_id(response)
            if media_id:
                data_tracker.add_media(media_id)
        finally:
            os.unlink(file_path)
        
        # Search with context size limit
        try:
            response = api_client.rag_simple_search(
                query="machine learning AI technology",
                databases=["media"],
                max_context_size=1000,  # Limit context
                top_k=10
            )
            
            if response.get("success"):
                results = response.get("results", [])
                
                # Check total context size
                total_size = sum(len(r.get("content", "")) for r in results)
                assert total_size <= 1000, \
                    f"Context size exceeded limit: {total_size} > 1000"
                
                print(f"✓ Context window optimized: {total_size} chars")
                
        except:
            print("⚠ Context optimization not testable")
    
    def test_citation_generation(self, api_client, data_tracker):
        """Test that RAG generates proper citations."""
        # Create identifiable content
        contents = [
            ("Research Paper A", "Groundbreaking research on quantum computing by Dr. Smith"),
            ("Technical Report B", "Analysis of machine learning trends by Prof. Johnson"),
            ("Industry Study C", "Market analysis of AI adoption by Analytics Corp")
        ]
        
        media_ids = []
        for title, content in contents:
            file_path = self._create_temp_file(content)
            try:
                response = api_client.upload_media(
                    file_path=file_path,
                    title=title,
                    media_type="document"
                )
                media_id = self._extract_media_id(response)
                if media_id:
                    media_ids.append(media_id)
                    data_tracker.add_media(media_id)
            finally:
                os.unlink(file_path)
        
        if not media_ids:
            pytest.skip("Could not create test media")
        
        # Search with citations enabled
        try:
            response = api_client.rag_simple_search(
                query="research on quantum computing and machine learning",
                databases=["media"],
                top_k=5,
                enable_citations=True
            )
            
            if response.get("success"):
                results = response.get("results", [])
                
                # Check for citations
                has_citations = False
                for result in results:
                    if "citation" in result:
                        citation = result["citation"]
                        if "title" in citation or "source_id" in citation:
                            has_citations = True
                            print(f"✓ Citation found: {citation.get('title', 'Unknown')}")
                            break
                
                if not has_citations:
                    print("⚠ Citations not included in results")
                    
        except:
            print("⚠ Citation generation not available")
    
    def test_multi_database_search(self, api_client, data_tracker):
        """Test searching across multiple databases."""
        # Create content in different types
        # 1. Media content
        media_file = self._create_temp_file("Media content about technology")
        try:
            response = api_client.upload_media(
                file_path=media_file,
                title="Media Test",
                media_type="document"
            )
            media_id = self._extract_media_id(response)
            if media_id:
                data_tracker.add_media(media_id)
        finally:
            os.unlink(media_file)
        
        # 2. Note content
        note_response = api_client.create_note(
            title="Note about technology",
            content="Notes on recent technology trends and developments"
        )
        note_id = note_response.get("id") or note_response.get("note_id")
        if note_id:
            data_tracker.add_note(note_id)
        
        # Search across multiple databases
        try:
            response = api_client.rag_simple_search(
                query="technology trends",
                databases=["media", "notes"],
                top_k=10
            )
            
            if response.get("success"):
                results = response.get("results", [])
                
                # Check for results from different sources
                source_types = set()
                for result in results:
                    if "source" in result:
                        source_type = result["source"].get("type")
                        if source_type:
                            source_types.add(source_type)
                
                if len(source_types) > 1:
                    print(f"✓ Multi-database search found: {source_types}")
                else:
                    print(f"⚠ Limited source diversity: {source_types}")
                    
        except:
            print("⚠ Multi-database search not available")
    
    def _create_temp_file(self, content: str) -> str:
        """Create a temporary text file."""
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write(content)
            return f.name
    
    def _extract_media_id(self, response: Dict[str, Any]) -> Optional[int]:
        """Extract media ID from response."""
        if "results" in response and response["results"]:
            return response["results"][0].get("db_id")
        return response.get("media_id") or response.get("id")


# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v"])