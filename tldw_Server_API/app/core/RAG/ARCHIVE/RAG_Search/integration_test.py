# integration_test.py - RAG Integration Testing
"""
Integration tests for the enhanced RAG components with existing endpoints.

Tests:
- Query expansion integration
- Reranking integration
- Caching integration
- Chunking integration
- End-to-end search flow
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import pytest
from loguru import logger

# Import new modules
from .advanced_query_expansion import AdvancedQueryExpander, ExpansionConfig, ExpansionStrategy
from .advanced_reranker import create_reranker, RerankingStrategy, RerankingConfig
from .enhanced_cache import (
    LRUCacheStrategy, SemanticCacheStrategy, TieredCacheStrategy,
    AdaptiveCacheStrategy, CacheManager
)
from .advanced_chunking import create_chunker, ChunkingStrategy

# Import existing modules
from .simplified.rag_service import SimplifiedRAGService
from .simplified.config import RAGConfig, SearchConfig, EmbeddingConfig
from .simplified.vector_store import SearchResult


class RAGIntegrationTester:
    """Test harness for RAG component integration"""
    
    def __init__(self, base_path: Optional[Path] = None):
        self.base_path = base_path or Path("./test_data")
        self.base_path.mkdir(exist_ok=True)
        
        # Initialize components
        self._init_components()
        
        # Test documents
        self.test_documents = [
            {
                "id": "doc1",
                "content": "Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
                "metadata": {"source": "intro", "type": "definition"}
            },
            {
                "id": "doc2",
                "content": "Deep learning uses neural networks with multiple layers to process complex patterns.",
                "metadata": {"source": "technical", "type": "explanation"}
            },
            {
                "id": "doc3",
                "content": "Natural language processing (NLP) helps computers understand human language.",
                "metadata": {"source": "nlp", "type": "definition"}
            },
            {
                "id": "doc4",
                "content": "RAG combines retrieval and generation for better AI responses using context from documents.",
                "metadata": {"source": "rag", "type": "technical"}
            },
            {
                "id": "doc5",
                "content": "LLMs like GPT use transformer architecture for text generation and understanding.",
                "metadata": {"source": "llm", "type": "technical"}
            }
        ]
    
    def _init_components(self):
        """Initialize all RAG components"""
        # Query expander
        self.query_expander = AdvancedQueryExpander(
            ExpansionConfig(
                strategies=[
                    ExpansionStrategy.LINGUISTIC,
                    ExpansionStrategy.ACRONYM,
                    ExpansionStrategy.DOMAIN
                ],
                max_expansions_per_strategy=2,
                total_max_expansions=5
            )
        )
        
        # Rerankers
        self.rerankers = {
            "cross_encoder": create_reranker(RerankingStrategy.CROSS_ENCODER, top_k=5),
            "diversity": create_reranker(RerankingStrategy.DIVERSITY, top_k=5),
            "multi_criteria": create_reranker(RerankingStrategy.MULTI_CRITERIA, top_k=5),
            "hybrid": create_reranker(RerankingStrategy.HYBRID, top_k=5)
        }
        
        # Cache strategies
        self.cache_manager = CacheManager([
            LRUCacheStrategy(max_size=100),
            TieredCacheStrategy(memory_size=50, disk_size=200, disk_path=self.base_path / "cache")
        ])
        
        # Chunkers
        self.chunkers = {
            "semantic": create_chunker(ChunkingStrategy.SEMANTIC, chunk_size=100),
            "structural": create_chunker(ChunkingStrategy.STRUCTURAL, chunk_size=100),
            "adaptive": create_chunker(ChunkingStrategy.ADAPTIVE, chunk_size=100)
        }
        
        # RAG service config
        self.rag_config = RAGConfig(
            search=SearchConfig(
                top_k=10,
                score_threshold=0.5,
                include_citations=True
            ),
            embeddings=EmbeddingConfig(
                model="all-MiniLM-L6-v2",
                dimension=384
            )
        )
    
    async def test_query_expansion(self) -> Dict[str, Any]:
        """Test query expansion integration"""
        logger.info("Testing query expansion...")
        
        test_queries = [
            "What is ML?",
            "How does RAG work?",
            "NLP applications",
            "deep learning models"
        ]
        
        results = {}
        for query in test_queries:
            expansions = await self.query_expander.expand_query(query)
            results[query] = {
                "original": query,
                "expansions": expansions,
                "count": len(expansions)
            }
            logger.info(f"Query: '{query}' -> {len(expansions)} expansions")
        
        return {
            "status": "success",
            "test": "query_expansion",
            "results": results
        }
    
    async def test_reranking(self) -> Dict[str, Any]:
        """Test reranking integration"""
        logger.info("Testing reranking...")
        
        # Simulate search results
        query = "machine learning applications"
        mock_results = [
            {"content": doc["content"], "metadata": doc["metadata"]}
            for doc in self.test_documents
        ]
        
        results = {}
        for name, reranker in self.rerankers.items():
            start_time = time.time()
            reranked = await reranker.rerank(query, mock_results)
            elapsed = time.time() - start_time
            
            results[name] = {
                "top_result": reranked[0].content[:50] + "..." if reranked else None,
                "num_results": len(reranked),
                "time_ms": round(elapsed * 1000, 2)
            }
            logger.info(f"Reranker '{name}': {len(reranked)} results in {elapsed:.3f}s")
        
        return {
            "status": "success",
            "test": "reranking",
            "query": query,
            "results": results
        }
    
    async def test_caching(self) -> Dict[str, Any]:
        """Test caching integration"""
        logger.info("Testing caching...")
        
        # Test data
        test_key = "test_query_1"
        test_value = {"results": ["result1", "result2"], "timestamp": time.time()}
        
        # Test set/get
        await self.cache_manager.set(test_key, test_value, query="test query")
        
        # Test hit
        start_time = time.time()
        cached_value = await self.cache_manager.get(test_key)
        hit_time = time.time() - start_time
        
        # Test miss
        start_time = time.time()
        missed_value = await self.cache_manager.get("nonexistent_key")
        miss_time = time.time() - start_time
        
        # Get stats
        stats = self.cache_manager.get_stats()
        
        return {
            "status": "success",
            "test": "caching",
            "results": {
                "cache_hit": cached_value is not None,
                "hit_time_ms": round(hit_time * 1000, 2),
                "miss_time_ms": round(miss_time * 1000, 2),
                "stats": {
                    name: {
                        "hits": s.hits,
                        "misses": s.misses,
                        "hit_rate": f"{s.hit_rate:.2%}"
                    }
                    for name, s in stats.items()
                }
            }
        }
    
    async def test_chunking(self) -> Dict[str, Any]:
        """Test chunking integration"""
        logger.info("Testing chunking...")
        
        # Test document with structure
        test_doc = """# Introduction to RAG

Retrieval-Augmented Generation (RAG) is a powerful technique that combines the strengths of retrieval systems with generative models.

## Key Components

1. **Retrieval System**: Searches for relevant documents
2. **Generation Model**: Creates responses based on retrieved context
3. **Integration Layer**: Combines retrieval and generation

### Benefits

- Improved accuracy with factual grounding
- Reduced hallucinations
- Dynamic knowledge updates

## Implementation

```python
def rag_pipeline(query):
    # Retrieve relevant documents
    docs = retriever.search(query)
    
    # Generate response with context
    response = generator.generate(query, context=docs)
    
    return response
```

This approach has revolutionized how we build AI systems."""
        
        results = {}
        for name, chunker in self.chunkers.items():
            chunks = chunker.chunk(test_doc)
            results[name] = {
                "num_chunks": len(chunks),
                "avg_size": sum(c.metadata.char_count for c in chunks) / len(chunks) if chunks else 0,
                "chunk_types": list(set(c.metadata.chunk_type for c in chunks)),
                "has_hierarchy": any(c.metadata.parent_id for c in chunks)
            }
            logger.info(f"Chunker '{name}': {len(chunks)} chunks")
        
        return {
            "status": "success",
            "test": "chunking",
            "doc_length": len(test_doc),
            "results": results
        }
    
    async def test_end_to_end_flow(self) -> Dict[str, Any]:
        """Test complete RAG flow with all components"""
        logger.info("Testing end-to-end RAG flow...")
        
        try:
            # 1. Initialize RAG service
            rag_service = SimplifiedRAGService(self.rag_config)
            
            # 2. Chunk and index documents
            logger.info("Indexing documents...")
            all_chunks = []
            for doc in self.test_documents:
                chunks = self.chunkers["adaptive"].chunk(doc["content"])
                for chunk in chunks:
                    all_chunks.append({
                        "id": f"{doc['id']}_{chunk.metadata.chunk_id}",
                        "content": chunk.text,
                        "metadata": {
                            **doc["metadata"],
                            "chunk_metadata": chunk.metadata.to_dict()
                        }
                    })
            
            # Index chunks
            await rag_service.index_documents(all_chunks)
            
            # 3. Test query with expansion
            original_query = "How does ML work?"
            expanded_queries = await self.query_expander.expand_query(original_query)
            all_queries = [original_query] + expanded_queries
            
            logger.info(f"Testing with {len(all_queries)} queries (original + expansions)")
            
            # 4. Search with caching
            search_results = []
            for query in all_queries[:3]:  # Limit to avoid too many searches
                # Check cache first
                cache_key = f"search_{query}"
                cached = await self.cache_manager.get(cache_key)
                
                if cached:
                    results = cached
                    logger.info(f"Cache hit for query: {query}")
                else:
                    # Perform search
                    results = await rag_service.search(query, top_k=5)
                    # Cache results
                    await self.cache_manager.set(cache_key, results, query=query)
                    logger.info(f"Searched and cached: {query}")
                
                search_results.extend(results)
            
            # 5. Deduplicate results
            unique_results = {}
            for result in search_results:
                if hasattr(result, 'id'):
                    unique_results[result.id] = result
                elif hasattr(result, 'document'):
                    # Use document content as key for deduplication
                    unique_results[result.document[:50]] = result
            
            # 6. Rerank results
            results_for_reranking = [
                {
                    "content": r.document if hasattr(r, 'document') else r.content,
                    "metadata": r.metadata if hasattr(r, 'metadata') else {}
                }
                for r in unique_results.values()
            ]
            
            reranked = await self.rerankers["hybrid"].rerank(
                original_query,
                results_for_reranking
            )
            
            # 7. Prepare final results
            final_results = []
            for i, result in enumerate(reranked[:5]):
                final_results.append({
                    "rank": i + 1,
                    "content": result.content[:100] + "...",
                    "relevance_score": result.relevance_score,
                    "rerank_score": result.rerank_score
                })
            
            return {
                "status": "success",
                "test": "end_to_end",
                "results": {
                    "original_query": original_query,
                    "num_expansions": len(expanded_queries),
                    "total_chunks_indexed": len(all_chunks),
                    "unique_results_before_rerank": len(unique_results),
                    "final_results": final_results,
                    "cache_stats": self.cache_manager.get_stats()
                }
            }
            
        except Exception as e:
            logger.error(f"End-to-end test failed: {e}")
            return {
                "status": "error",
                "test": "end_to_end",
                "error": str(e)
            }
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration tests"""
        logger.info("Starting RAG integration tests...")
        
        results = {
            "timestamp": time.time(),
            "tests": {}
        }
        
        # Run tests
        tests = [
            ("query_expansion", self.test_query_expansion),
            ("reranking", self.test_reranking),
            ("caching", self.test_caching),
            ("chunking", self.test_chunking),
            ("end_to_end", self.test_end_to_end_flow)
        ]
        
        for test_name, test_func in tests:
            try:
                result = await test_func()
                results["tests"][test_name] = result
            except Exception as e:
                logger.error(f"Test {test_name} failed: {e}")
                results["tests"][test_name] = {
                    "status": "error",
                    "error": str(e)
                }
        
        # Summary
        passed = sum(1 for t in results["tests"].values() if t.get("status") == "success")
        total = len(results["tests"])
        results["summary"] = {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "success_rate": f"{(passed/total)*100:.1f}%" if total > 0 else "0%"
        }
        
        return results


# Pytest fixtures and tests
@pytest.fixture
async def rag_tester():
    """Create RAG integration tester"""
    return RAGIntegrationTester()


@pytest.mark.asyncio
async def test_query_expansion_integration(rag_tester):
    """Test query expansion component"""
    result = await rag_tester.test_query_expansion()
    assert result["status"] == "success"
    assert len(result["results"]) > 0


@pytest.mark.asyncio
async def test_reranking_integration(rag_tester):
    """Test reranking component"""
    result = await rag_tester.test_reranking()
    assert result["status"] == "success"
    assert all(r["num_results"] > 0 for r in result["results"].values())


@pytest.mark.asyncio
async def test_caching_integration(rag_tester):
    """Test caching component"""
    result = await rag_tester.test_caching()
    assert result["status"] == "success"
    assert result["results"]["cache_hit"] is True


@pytest.mark.asyncio
async def test_chunking_integration(rag_tester):
    """Test chunking component"""
    result = await rag_tester.test_chunking()
    assert result["status"] == "success"
    assert all(r["num_chunks"] > 0 for r in result["results"].values())


@pytest.mark.asyncio
async def test_end_to_end_flow(rag_tester):
    """Test complete RAG flow"""
    result = await rag_tester.test_end_to_end_flow()
    assert result["status"] == "success"
    assert len(result["results"]["final_results"]) > 0


# CLI runner
async def main():
    """Run integration tests from command line"""
    tester = RAGIntegrationTester()
    results = await tester.run_all_tests()
    
    # Pretty print results
    print("\n" + "="*60)
    print("RAG Integration Test Results")
    print("="*60)
    
    for test_name, test_result in results["tests"].items():
        status = test_result.get("status", "unknown")
        symbol = "✓" if status == "success" else "✗"
        print(f"\n{symbol} {test_name.upper()}: {status}")
        
        if status == "error":
            print(f"  Error: {test_result.get('error', 'Unknown error')}")
        elif test_name == "end_to_end" and status == "success":
            # Show end-to-end details
            details = test_result["results"]
            print(f"  - Query: {details['original_query']}")
            print(f"  - Expansions: {details['num_expansions']}")
            print(f"  - Chunks indexed: {details['total_chunks_indexed']}")
            print(f"  - Final results: {len(details['final_results'])}")
    
    print(f"\n{results['summary']}")
    print("="*60)


if __name__ == "__main__":
    # Handle async execution properly - check if event loop is already running
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're here, a loop is already running (e.g., in Jupyter or async context)
        # Create a task instead of using asyncio.run()
        loop.create_task(main())
    except RuntimeError:
        # No event loop is running, so we can use asyncio.run()
        asyncio.run(main())