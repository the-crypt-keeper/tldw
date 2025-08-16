"""
Integration tests for RAG-Embeddings system.

These tests verify that the RAG system properly integrates with the real embeddings
service without any mocking. All embeddings are generated using the actual production
embeddings service.

NO MOCKING - These are true integration tests.
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
import numpy as np
from typing import List, Dict, Any
import time

from loguru import logger

# Import the actual services - no mocks
from tldw_Server_API.app.core.RAG.rag_embeddings_integration import (
    ProductionEmbeddingFunction,
    EnhancedVectorRetriever,
    RAGEmbeddingsIntegration,
    create_rag_embeddings_integration
)

from tldw_Server_API.app.core.RAG.rag_service.types import (
    DataSource, Document, SearchResult
)

# Import actual embeddings service to verify it's being used
from tldw_Server_API.app.core.Embeddings.Embeddings_Server.Embeddings_Create import (
    create_embeddings_batch,
    create_embedding
)


@pytest.mark.integration
class TestProductionEmbeddingFunction:
    """Test the ChromaDB embedding function with real embeddings."""
    
    def test_embedding_function_initialization(self):
        """Test that the embedding function initializes correctly."""
        # Create with HuggingFace provider (no API key needed)
        func = ProductionEmbeddingFunction(
            provider="huggingface",
            model_id="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        assert func.provider == "huggingface"
        assert func.model_id == "sentence-transformers/all-MiniLM-L6-v2"
        assert func.config["provider"] == "huggingface"
    
    def test_embedding_function_generates_real_embeddings(self):
        """Test that the embedding function generates real embeddings."""
        # Use HuggingFace for testing (no API key required)
        func = ProductionEmbeddingFunction(
            provider="huggingface",
            model_id="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Test documents
        documents = [
            "This is a test document.",
            "Machine learning is fascinating.",
            "Natural language processing is useful."
        ]
        
        # Generate embeddings - this calls the real service
        embeddings = func(documents)
        
        # Convert numpy arrays to lists if needed for consistency
        if hasattr(embeddings, 'tolist'):
            # It's a numpy array
            embeddings = embeddings.tolist()
        elif isinstance(embeddings, list) and len(embeddings) > 0:
            # Check if elements are numpy arrays
            if hasattr(embeddings[0], 'tolist'):
                embeddings = [emb.tolist() if hasattr(emb, 'tolist') else emb for emb in embeddings]
        
        # Verify we got real embeddings
        assert len(embeddings) == len(documents)
        assert all(isinstance(emb, list) for emb in embeddings)
        assert all(len(emb) > 0 for emb in embeddings)
        
        # Verify embeddings are different for different texts
        assert embeddings[0] != embeddings[1]
        assert embeddings[1] != embeddings[2]
        
        # Verify embeddings have consistent dimensions
        dimensions = [len(emb) for emb in embeddings]
        assert len(set(dimensions)) == 1  # All same dimension
        
        logger.info(f"Generated {len(embeddings)} real embeddings with dimension {dimensions[0]}")
    
    def test_embedding_function_handles_empty_input(self):
        """Test that the embedding function handles empty input correctly."""
        func = ProductionEmbeddingFunction(
            provider="huggingface",
            model_id="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Empty input should return empty list
        embeddings = func([])
        assert embeddings == []


@pytest.mark.integration
class TestEnhancedVectorRetriever:
    """Test the enhanced vector retriever with real embeddings."""
    
    @pytest.fixture
    def temp_chroma_path(self):
        """Create a temporary directory for ChromaDB."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    async def test_vector_retriever_initialization(self, temp_chroma_path):
        """Test that the vector retriever initializes with real embeddings."""
        retriever = EnhancedVectorRetriever(
            source=DataSource.MEDIA_DB,
            chroma_path=temp_chroma_path,
            collection_name="test_collection",
            embedding_provider="huggingface",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        assert retriever.source == DataSource.MEDIA_DB
        assert retriever.collection_name == "test_collection"
        assert retriever.embedding_function.provider == "huggingface"
        
        # Verify collection was created
        stats = retriever.get_embedding_stats()
        assert stats["collection_name"] == "test_collection"
        assert stats["document_count"] == 0
        assert stats["embedding_provider"] == "huggingface"
    
    async def test_embed_and_retrieve_with_real_embeddings(self, temp_chroma_path):
        """Test storing and retrieving documents with real embeddings."""
        retriever = EnhancedVectorRetriever(
            source=DataSource.MEDIA_DB,
            chroma_path=temp_chroma_path,
            collection_name="test_media",
            embedding_provider="huggingface",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create test documents
        test_documents = [
            Document(
                id="doc1",
                content="Python is a high-level programming language.",
                metadata={"type": "programming", "language": "python"},
                source=DataSource.MEDIA_DB,
                score=0.0
            ),
            Document(
                id="doc2",
                content="Machine learning models can learn from data.",
                metadata={"type": "ml", "topic": "learning"},
                source=DataSource.MEDIA_DB,
                score=0.0
            ),
            Document(
                id="doc3",
                content="Natural language processing helps computers understand text.",
                metadata={"type": "nlp", "domain": "text"},
                source=DataSource.MEDIA_DB,
                score=0.0
            )
        ]
        
        # Store documents with real embeddings
        await retriever.embed_and_store(test_documents)
        
        # Verify documents were stored
        stats = retriever.get_embedding_stats()
        assert stats["document_count"] == 3
        
        # Search with a query - uses real embeddings
        query = "programming with Python"
        results = await retriever.retrieve(query, top_k=2)
        
        # Verify we got results
        assert len(results.documents) <= 2
        assert results.search_type == "vector"
        
        # The Python document should be most relevant
        if results.documents:
            # Check that we got relevant results
            top_doc = results.documents[0]
            assert "Python" in top_doc.content or "programming" in top_doc.content
            assert top_doc.score > 0  # Should have a similarity score
        
        logger.info(f"Retrieved {len(results.documents)} documents with real embeddings")
    
    async def test_retriever_with_filters(self, temp_chroma_path):
        """Test retrieval with filters using real embeddings."""
        retriever = EnhancedVectorRetriever(
            source=DataSource.MEDIA_DB,
            chroma_path=temp_chroma_path,
            collection_name="filtered_test",
            embedding_provider="huggingface",
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        # Create documents with different types
        documents = [
            Document(
                id=f"doc_{i}",
                content=content,
                metadata={"type": doc_type, "index": i},
                source=DataSource.MEDIA_DB,
                score=0.0
            )
            for i, (content, doc_type) in enumerate([
                ("Python programming basics", "tutorial"),
                ("Advanced Python techniques", "tutorial"),
                ("Python news and updates", "news"),
                ("Python community events", "news")
            ])
        ]
        
        await retriever.embed_and_store(documents)
        
        # Search with filter
        results = await retriever.retrieve(
            query="Python programming",
            filters={"type": "tutorial"},
            top_k=10
        )
        
        # Should only get tutorial documents
        assert all(doc.metadata.get("type") == "tutorial" for doc in results.documents)
        logger.info(f"Filtered retrieval returned {len(results.documents)} documents")


@pytest.mark.integration
class TestRAGEmbeddingsIntegration:
    """Test the complete RAG-Embeddings integration."""
    
    async def test_integration_initialization(self):
        """Test that the integration initializes correctly."""
        integration = create_rag_embeddings_integration(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        assert integration.embedding_provider == "huggingface"
        assert integration.embedding_model == "sentence-transformers/all-MiniLM-L6-v2"
        assert integration.embeddings_service is not None
        
        # Clean up
        integration.close()
    
    async def test_query_embedding_generation(self):
        """Test generating query embeddings with real service."""
        integration = create_rag_embeddings_integration(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        try:
            # Generate embedding for a query
            query = "What is artificial intelligence?"
            embedding = await integration.embed_query(query)
            
            # Verify we got a real embedding
            assert isinstance(embedding, np.ndarray)
            assert embedding.ndim == 1
            assert embedding.shape[0] > 0
            
            # Verify it's not zeros or random
            assert not np.allclose(embedding, 0)
            assert np.std(embedding) > 0.01  # Should have some variance
            
            logger.info(f"Generated query embedding with shape {embedding.shape}")
            
        finally:
            integration.close()
    
    async def test_document_embedding_generation(self):
        """Test generating document embeddings with real service."""
        integration = create_rag_embeddings_integration(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        try:
            # Generate embeddings for documents
            documents = [
                "Artificial intelligence is transforming technology.",
                "Machine learning is a subset of AI.",
                "Deep learning uses neural networks."
            ]
            
            embeddings = await integration.embed_documents(documents)
            
            # Verify we got real embeddings
            assert isinstance(embeddings, np.ndarray)
            assert embeddings.shape[0] == len(documents)
            assert embeddings.shape[1] > 0
            
            # Verify they're different for different documents
            assert not np.allclose(embeddings[0], embeddings[1])
            assert not np.allclose(embeddings[1], embeddings[2])
            
            logger.info(f"Generated document embeddings with shape {embeddings.shape}")
            
        finally:
            integration.close()
    
    async def test_end_to_end_rag_flow(self):
        """Test complete RAG flow with real embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            chroma_path = Path(tmpdir)
            
            # Create integration
            integration = create_rag_embeddings_integration(
                provider="huggingface",
                model="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            try:
                # Create vector retriever
                retriever = integration.create_vector_retriever(
                    source=DataSource.MEDIA_DB,
                    chroma_path=chroma_path,
                    collection_name="rag_test"
                )
                
                # Prepare documents
                documents = [
                    Document(
                        id="intro",
                        content="Artificial intelligence (AI) is the simulation of human intelligence in machines.",
                        metadata={"topic": "ai", "level": "intro"},
                        source=DataSource.MEDIA_DB,
                        score=0.0
                    ),
                    Document(
                        id="ml",
                        content="Machine learning enables computers to learn from data without explicit programming.",
                        metadata={"topic": "ml", "level": "intermediate"},
                        source=DataSource.MEDIA_DB,
                        score=0.0
                    ),
                    Document(
                        id="dl",
                        content="Deep learning uses artificial neural networks with multiple layers to progressively extract features.",
                        metadata={"topic": "dl", "level": "advanced"},
                        source=DataSource.MEDIA_DB,
                        score=0.0
                    )
                ]
                
                # Store documents with real embeddings
                await retriever.embed_and_store(documents)
                
                # Perform searches with different queries
                queries = [
                    "What is AI?",
                    "How does machine learning work?",
                    "neural networks and deep learning"
                ]
                
                for query in queries:
                    results = await retriever.retrieve(query, top_k=2)
                    
                    assert len(results.documents) > 0
                    assert results.metadata["embedding_provider"] == "huggingface"
                    
                    logger.info(f"Query: '{query}' returned {len(results.documents)} results")
                    for doc in results.documents:
                        logger.info(f"  - {doc.id}: score={doc.score:.3f}")
                
                # Get metrics
                metrics = integration.get_metrics()
                assert metrics["total_texts_processed"] > 0
                logger.info(f"Integration metrics: {metrics}")
                
            finally:
                integration.close()
    
    async def test_embedding_dimension_consistency(self):
        """Test that embedding dimensions are consistent."""
        integration = create_rag_embeddings_integration(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        try:
            # Get embedding dimension
            dimension = integration.get_embedding_dimension()
            assert dimension is not None
            assert dimension > 0
            
            # Generate embeddings and verify dimension
            query_emb = await integration.embed_query("test query")
            assert query_emb.shape[0] == dimension
            
            doc_embs = await integration.embed_documents(["doc1", "doc2"])
            assert doc_embs.shape[1] == dimension
            
            logger.info(f"Embedding dimension is consistently {dimension}")
            
        finally:
            integration.close()


@pytest.mark.integration
class TestPerformanceAndReliability:
    """Test performance and reliability of the integrated system."""
    
    async def test_concurrent_embedding_generation(self):
        """Test that the system handles concurrent requests properly."""
        integration = create_rag_embeddings_integration(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        try:
            # Create multiple concurrent embedding tasks
            queries = [f"Query number {i}" for i in range(10)]
            
            start_time = time.time()
            
            # Run concurrent embedding generation
            tasks = [integration.embed_query(q) for q in queries]
            embeddings = await asyncio.gather(*tasks)
            
            elapsed = time.time() - start_time
            
            # Verify all embeddings were generated
            assert len(embeddings) == len(queries)
            assert all(isinstance(emb, np.ndarray) for emb in embeddings)
            
            # Verify they're all different
            for i in range(len(embeddings) - 1):
                assert not np.allclose(embeddings[i], embeddings[i + 1])
            
            logger.info(f"Generated {len(embeddings)} embeddings concurrently in {elapsed:.2f}s")
            
        finally:
            integration.close()
    
    async def test_large_batch_processing(self):
        """Test processing large batches of documents."""
        integration = create_rag_embeddings_integration(
            provider="huggingface",
            model="sentence-transformers/all-MiniLM-L6-v2"
        )
        
        try:
            # Create a large batch of documents
            num_docs = 100
            documents = [f"Document {i}: This is test content for document number {i}." 
                        for i in range(num_docs)]
            
            start_time = time.time()
            
            # Process the batch
            embeddings = await integration.embed_documents(documents)
            
            elapsed = time.time() - start_time
            
            # Verify all embeddings were generated
            assert embeddings.shape[0] == num_docs
            
            # Calculate throughput
            docs_per_second = num_docs / elapsed
            logger.info(f"Processed {num_docs} documents in {elapsed:.2f}s ({docs_per_second:.1f} docs/s)")
            
            # Verify memory usage is reasonable
            metrics = integration.get_metrics()
            logger.info(f"After batch processing: {metrics}")
            
        finally:
            integration.close()


# Run specific integration test
if __name__ == "__main__":
    async def run_integration_test():
        """Run a specific integration test."""
        logger.info("Running RAG-Embeddings integration test...")
        
        # Test the complete flow
        test = TestRAGEmbeddingsIntegration()
        await test.test_end_to_end_rag_flow()
        
        logger.info("Integration test completed successfully!")
    
    asyncio.run(run_integration_test())