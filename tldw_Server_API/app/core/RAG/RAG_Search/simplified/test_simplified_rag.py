#!/usr/bin/env python3
"""
Test script for the simplified RAG implementation.

This script tests the basic functionality of the simplified RAG service
including indexing, searching, and citations.
"""

import asyncio
import sys
from pathlib import Path
import logging
import json

# Add the project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from tldw_chatbook.RAG_Search.simplified import (
    RAGService, 
    RAGConfig,
    create_config_for_testing,
    Citation,
    CitationType
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def test_basic_functionality():
    """Test basic RAG functionality."""
    print("\n" + "="*60)
    print("Testing Simplified RAG Implementation")
    print("="*60 + "\n")
    
    # 1. Create test configuration
    print("1. Creating test configuration...")
    config = create_config_for_testing(use_memory_store=True)
    print(f"   - Embedding model: {config.embedding_model}")
    print(f"   - Vector store: {config.vector_store_type}")
    print(f"   - Chunk size: {config.chunk_size}")
    
    # 2. Initialize RAG service
    print("\n2. Initializing RAG service...")
    try:
        rag_service = RAGService(config)
        print("   ✓ RAG service initialized successfully")
    except Exception as e:
        print(f"   ✗ Failed to initialize: {e}")
        return
    
    # 3. Test documents
    test_documents = [
        {
            "id": "doc1",
            "title": "Introduction to RAG",
            "content": """
            Retrieval-Augmented Generation (RAG) is a technique that combines 
            information retrieval with text generation. It allows AI models to 
            access external knowledge bases to provide more accurate and up-to-date 
            responses. RAG systems typically involve three main components: 
            a retriever that finds relevant documents, a reader that extracts 
            information, and a generator that produces the final response.
            
            The key advantage of RAG is that it grounds AI responses in real data,
            reducing hallucinations and improving factual accuracy. This makes RAG
            particularly useful for question-answering systems, chatbots, and
            knowledge management applications.
            """,
            "metadata": {
                "author": "AI Research Team",
                "date": "2024-01-15",
                "category": "AI/ML"
            }
        },
        {
            "id": "doc2", 
            "title": "Vector Databases Explained",
            "content": """
            Vector databases are specialized systems designed to store and search
            high-dimensional vector embeddings. Unlike traditional databases that
            use exact matches, vector databases find similar items based on 
            distance metrics like cosine similarity.
            
            Popular vector databases include ChromaDB, Pinecone, and Weaviate.
            These systems enable semantic search, where queries return results
            based on meaning rather than keyword matches. This is essential for
            modern AI applications like RAG systems, recommendation engines, and
            similarity search.
            """,
            "metadata": {
                "author": "Database Expert",
                "date": "2024-02-20",
                "category": "Databases"
            }
        },
        {
            "id": "doc3",
            "title": "Building Production RAG Systems", 
            "content": """
            When building production RAG systems, several factors must be considered:
            performance, scalability, accuracy, and cost. Key design decisions include
            choosing the right embedding model, vector database, and chunking strategy.
            
            Performance optimization techniques include caching embeddings, using
            appropriate chunk sizes, and implementing efficient retrieval algorithms.
            For scalability, consider distributed vector stores and load balancing.
            Accuracy can be improved through hybrid search combining semantic and
            keyword matching, as well as re-ranking techniques.
            """,
            "metadata": {
                "author": "Engineering Team",
                "date": "2024-03-10",
                "category": "Engineering"
            }
        }
    ]
    
    # 4. Index documents
    print("\n3. Indexing test documents...")
    for doc in test_documents:
        result = await rag_service.index_document(
            doc_id=doc["id"],
            content=doc["content"],
            title=doc["title"],
            metadata=doc["metadata"]
        )
        if result.success:
            print(f"   ✓ Indexed '{doc['title']}' ({result.chunks_created} chunks)")
        else:
            print(f"   ✗ Failed to index '{doc['title']}': {result.error}")
    
    # 5. Get metrics
    print("\n4. Checking metrics...")
    metrics = rag_service.get_metrics()
    print(f"   - Documents indexed: {metrics['service_metrics']['documents_indexed']}")
    print(f"   - Total chunks: {metrics['service_metrics']['total_chunks_created']}")
    print(f"   - Embedding dimension: {metrics['embeddings_metrics'].get('embedding_dimension', 'Unknown')}")
    
    # 6. Test searches
    test_queries = [
        "What is RAG and how does it work?",
        "Tell me about vector databases",
        "How to optimize RAG performance?",
        "What are the components of a RAG system?"
    ]
    
    print("\n5. Testing semantic search with citations...")
    for query in test_queries:
        print(f"\n   Query: '{query}'")
        results = await rag_service.search(
            query=query,
            top_k=3,
            search_type="semantic",
            include_citations=True
        )
        
        if results:
            for i, result in enumerate(results, 1):
                print(f"\n   Result {i} (score: {result.score:.3f}):")
                print(f"   - Document: {result.metadata.get('doc_title', 'Unknown')}")
                print(f"   - Text preview: {result.document[:100]}...")
                
                if hasattr(result, 'citations') and result.citations:
                    print(f"   - Citations ({len(result.citations)}):")
                    for citation in result.citations:
                        print(f"     • {citation.format_citation('inline')}")
                        print(f"       Confidence: {citation.confidence:.3f}")
        else:
            print("   No results found")
    
    # 7. Test citation features
    print("\n6. Testing citation features...")
    if results and hasattr(results[0], 'citations'):
        result = results[0]
        print(f"   - Unique sources: {result.get_unique_sources()}")
        print(f"   - Formatted with citations: ")
        print(f"     {result.format_with_citations('inline', max_citations=2)[:200]}...")
        
        # Test citation serialization
        citation = result.citations[0]
        citation_dict = citation.to_dict()
        print(f"\n   - Citation as dict: {json.dumps(citation_dict, indent=2)}")
    
    # 8. Clean up
    print("\n7. Cleaning up...")
    rag_service.clear_index()
    rag_service.close()
    print("   ✓ Cleanup complete")
    
    print("\n" + "="*60)
    print("Test completed successfully!")
    print("="*60 + "\n")


async def test_error_handling():
    """Test error handling and edge cases."""
    print("\n" + "="*60)
    print("Testing Error Handling")
    print("="*60 + "\n")
    
    config = create_config_for_testing()
    rag_service = RAGService(config)
    
    # Test empty document
    print("1. Testing empty document...")
    result = await rag_service.index_document("empty", "")
    print(f"   - Success: {result.success}, Chunks: {result.chunks_created}")
    
    # Test search with no documents
    print("\n2. Testing search with no indexed documents...")
    results = await rag_service.search("test query")
    print(f"   - Results: {len(results)}")
    
    # Test invalid configuration
    print("\n3. Testing configuration validation...")
    test_config = RAGConfig()
    test_config.chunking.chunk_overlap = 500  # Greater than chunk_size
    errors = test_config.validate()
    print(f"   - Validation errors: {errors}")
    
    rag_service.close()


async def test_different_models():
    """Test with different embedding models if available."""
    print("\n" + "="*60)
    print("Testing Different Embedding Models")
    print("="*60 + "\n")
    
    models_to_test = [
        "sentence-transformers/all-MiniLM-L6-v2",
        # Add more models if you have them available
    ]
    
    for model in models_to_test:
        print(f"\nTesting with model: {model}")
        try:
            config = create_config_for_testing()
            config.embedding.model = model
            
            rag_service = RAGService(config)
            
            # Quick test
            result = await rag_service.index_document(
                "test_doc",
                "This is a test document for embedding model testing."
            )
            
            if result.success:
                print(f"   ✓ Successfully indexed with {model}")
                
                # Test search
                results = await rag_service.search("test document")
                print(f"   ✓ Search returned {len(results)} results")
            else:
                print(f"   ✗ Failed to index: {result.error}")
            
            rag_service.close()
            
        except Exception as e:
            print(f"   ✗ Error with model {model}: {e}")


async def main():
    """Run all tests."""
    try:
        await test_basic_functionality()
        await test_error_handling()
        await test_different_models()
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())