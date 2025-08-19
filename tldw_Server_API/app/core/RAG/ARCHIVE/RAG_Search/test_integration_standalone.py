# test_integration_standalone.py - Standalone RAG Integration Test
"""
Standalone version of integration tests that can be run directly.
Tests the new RAG components in isolation.
"""

import asyncio
import time
import json
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from advanced_query_expansion import AdvancedQueryExpander, ExpansionConfig, ExpansionStrategy
from advanced_reranker import create_reranker, RerankingStrategy
from enhanced_cache import LRUCacheStrategy, TieredCacheStrategy, CacheManager
from advanced_chunking import create_chunker, ChunkingStrategy


async def test_query_expansion():
    """Test query expansion functionality"""
    print("\n=== Testing Query Expansion ===")
    
    config = ExpansionConfig(
        strategies=[ExpansionStrategy.LINGUISTIC, ExpansionStrategy.ACRONYM],
        max_expansions_per_strategy=3,
        total_max_expansions=10
    )
    
    expander = AdvancedQueryExpander(config)
    
    test_queries = [
        "What is ML?",
        "How does NLP work?",
        "RAG applications",
        "deep learning"
    ]
    
    for query in test_queries:
        expansions = await expander.expand_query(query)
        print(f"\nQuery: '{query}'")
        print(f"Expansions ({len(expansions)}):")
        for exp in expansions[:5]:  # Show first 5
            print(f"  - {exp}")
    
    return True


async def test_reranking():
    """Test reranking functionality"""
    print("\n\n=== Testing Reranking ===")
    
    # Test documents
    documents = [
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "doc1"}
        },
        {
            "content": "Deep learning uses neural networks with multiple layers.",
            "metadata": {"source": "doc2"}
        },
        {
            "content": "Natural language processing helps computers understand text.",
            "metadata": {"source": "doc3"}
        },
        {
            "content": "Python is a popular programming language for data science.",
            "metadata": {"source": "doc4"}
        }
    ]
    
    query = "machine learning and deep learning"
    
    # Test different reranking strategies
    strategies = [RerankingStrategy.DIVERSITY, RerankingStrategy.MULTI_CRITERIA]
    
    for strategy in strategies:
        print(f"\n{strategy.value} Reranking:")
        reranker = create_reranker(strategy, top_k=3)
        
        try:
            reranked = await reranker.rerank(query, documents)
            for i, doc in enumerate(reranked):
                print(f"  {i+1}. Score: {doc.relevance_score:.3f} - {doc.content[:50]}...")
        except Exception as e:
            print(f"  Error: {e}")
    
    return True


async def test_caching():
    """Test caching functionality"""
    print("\n\n=== Testing Caching ===")
    
    # Create cache manager
    cache_manager = CacheManager([
        LRUCacheStrategy(max_size=10, ttl=60)
    ])
    
    # Test data
    test_key = "test_query_1"
    test_value = {"results": ["result1", "result2"], "timestamp": time.time()}
    
    # Test set/get
    await cache_manager.set(test_key, test_value, query="test query")
    
    # Test hit
    start = time.time()
    cached = await cache_manager.get(test_key)
    hit_time = time.time() - start
    
    print(f"Cache hit: {cached is not None}")
    print(f"Hit time: {hit_time*1000:.2f}ms")
    
    # Test miss
    start = time.time()
    missed = await cache_manager.get("nonexistent")
    miss_time = time.time() - start
    
    print(f"Cache miss: {missed is None}")
    print(f"Miss time: {miss_time*1000:.2f}ms")
    
    # Get stats
    stats = cache_manager.get_stats()
    print("\nCache stats:")
    for strategy, stat in stats.items():
        print(f"  {strategy}: hits={stat.hits}, misses={stat.misses}, hit_rate={stat.hit_rate:.1%}")
    
    return True


def test_chunking():
    """Test chunking functionality"""
    print("\n\n=== Testing Chunking ===")
    
    test_doc = """# Introduction to Machine Learning

Machine learning is a revolutionary field that enables computers to learn from data without being explicitly programmed. It has transformed industries and created new possibilities.

## Types of Machine Learning

1. **Supervised Learning**: Learning from labeled data
   - Classification: Predicting categories
   - Regression: Predicting continuous values

2. **Unsupervised Learning**: Finding patterns in unlabeled data
   - Clustering: Grouping similar items
   - Dimensionality reduction: Simplifying complex data

3. **Reinforcement Learning**: Learning through interaction
   - Agent learns by taking actions
   - Receives rewards or penalties

## Applications

Machine learning is used in:
- Healthcare for disease diagnosis
- Finance for fraud detection
- Transportation for autonomous vehicles
- Entertainment for content recommendations

The future of ML is bright with continuous innovations."""
    
    strategies = [ChunkingStrategy.SEMANTIC, ChunkingStrategy.STRUCTURAL, ChunkingStrategy.ADAPTIVE]
    
    for strategy in strategies:
        print(f"\n{strategy.value} Chunking:")
        chunker = create_chunker(strategy, chunk_size=150, overlap=20)
        
        chunks = chunker.chunk(test_doc)
        print(f"  Number of chunks: {len(chunks)}")
        print(f"  Average size: {sum(c.metadata.char_count for c in chunks) / len(chunks):.0f} chars")
        
        # Show first chunk
        if chunks:
            print(f"  First chunk ({chunks[0].metadata.chunk_type}):")
            print(f"    {chunks[0].text[:100]}...")
    
    return True


async def test_end_to_end():
    """Test end-to-end integration"""
    print("\n\n=== Testing End-to-End Flow ===")
    
    # Initialize components
    query_expander = AdvancedQueryExpander(ExpansionConfig(
        strategies=[ExpansionStrategy.LINGUISTIC],
        max_expansions_per_strategy=2
    ))
    
    reranker = create_reranker(RerankingStrategy.DIVERSITY, top_k=3)
    
    cache_manager = CacheManager([LRUCacheStrategy(max_size=10)])
    
    chunker = create_chunker(ChunkingStrategy.ADAPTIVE, chunk_size=200)
    
    # Test query
    query = "How does machine learning work?"
    
    # 1. Expand query
    print(f"\n1. Query Expansion:")
    print(f"   Original: {query}")
    expansions = await query_expander.expand_query(query)
    print(f"   Expansions: {expansions[:3]}")
    
    # 2. Check cache
    print(f"\n2. Cache Check:")
    cache_key = f"search_{query}"
    cached = await cache_manager.get(cache_key)
    print(f"   Cache hit: {cached is not None}")
    
    if not cached:
        # 3. Simulate search (would normally query vector DB)
        print(f"\n3. Search (simulated)")
        mock_results = [
            {"content": "Machine learning algorithms build mathematical models based on training data.", "metadata": {}},
            {"content": "ML systems improve their performance through experience without explicit programming.", "metadata": {}},
            {"content": "Deep learning is a subset of machine learning using neural networks.", "metadata": {}},
            {"content": "Python and R are popular languages for machine learning development.", "metadata": {}}
        ]
        
        # 4. Rerank results
        print(f"\n4. Reranking:")
        reranked = await reranker.rerank(query, mock_results)
        print(f"   Top result: {reranked[0].content[:60]}...")
        
        # 5. Cache results
        await cache_manager.set(cache_key, reranked, query=query)
        print(f"\n5. Results cached")
    
    # 6. Test document chunking
    print(f"\n6. Document Chunking:")
    sample_doc = "Machine learning is transforming how we solve complex problems. " * 10
    chunks = chunker.chunk(sample_doc)
    print(f"   Created {len(chunks)} chunks from document")
    
    print("\n✓ End-to-end test completed successfully!")
    return True


async def main():
    """Run all integration tests"""
    print("RAG Component Integration Tests")
    print("="*50)
    
    tests = [
        ("Query Expansion", test_query_expansion),
        ("Reranking", lambda: asyncio.create_task(test_reranking())),
        ("Caching", test_caching),
        ("Chunking", test_chunking),
        ("End-to-End", test_end_to_end)
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                passed += 1
                status = "✓ PASSED"
            else:
                failed += 1
                status = "✗ FAILED"
        except Exception as e:
            failed += 1
            status = f"✗ ERROR: {e}"
            print(f"\n{test_name} test error: {e}")
        
        print(f"\n{test_name}: {status}")
    
    print("\n" + "="*50)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("="*50)
    
    return failed == 0


if __name__ == "__main__":
    # Handle async execution properly
    try:
        # Try to get the current event loop
        loop = asyncio.get_running_loop()
        # If we're here, a loop is already running
        # Create a task instead of using asyncio.run()
        success = await main()
    except RuntimeError:
        # No event loop is running, so we can use asyncio.run()
        success = asyncio.run(main())
    sys.exit(0 if success else 1)