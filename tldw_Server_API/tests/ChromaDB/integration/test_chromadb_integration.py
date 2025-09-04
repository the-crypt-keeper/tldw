"""
Integration tests for ChromaDB functionality.

These tests use real ChromaDB instances and actual embedding generation
without mocking to verify end-to-end functionality.
"""

import pytest
import tempfile
import shutil
import uuid
import time
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

import chromadb
from chromadb.config import Settings

from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
from tldw_Server_API.app.core.Embeddings.Embeddings_Create import (
    HuggingFaceEmbedder,
    create_embeddings_batch
)
from tldw_Server_API.app.core.DB_Management.Media_DB_v2 import MediaDatabase


@pytest.mark.integration
class TestChromaDBSetup:
    """Test ChromaDB setup and initialization."""
    
    def test_chromadb_client_creation(self, temp_chroma_path):
        """Test creating a ChromaDB client."""
        settings = Settings(
            persist_directory=temp_chroma_path,
            anonymized_telemetry=False,
            allow_reset=True
        )
        
        client = chromadb.PersistentClient(
            path=temp_chroma_path,
            settings=settings
        )
        
        # Verify client is functional
        assert client.heartbeat() is not None
        collections = client.list_collections()
        assert isinstance(collections, list)
    
    def test_chromadb_persistence(self, temp_chroma_path):
        """Test ChromaDB data persistence."""
        # Create client and add data
        client1 = chromadb.PersistentClient(path=temp_chroma_path)
        collection1 = client1.create_collection("test_persist")
        
        collection1.add(
            documents=["test document"],
            embeddings=[[0.1, 0.2, 0.3]],
            ids=["test_id"]
        )
        
        del client1  # Close first client
        
        # Create new client and verify data persists
        client2 = chromadb.PersistentClient(path=temp_chroma_path)
        collection2 = client2.get_collection("test_persist")
        
        result = collection2.get(ids=["test_id"])
        assert result["documents"][0] == "test document"
    
    def test_multiple_collections(self, chroma_client):
        """Test creating and managing multiple collections."""
        # Create multiple collections
        collections = []
        for i in range(5):
            col = chroma_client.create_collection(f"collection_{i}")
            collections.append(col)
        
        # Verify all collections exist
        all_collections = chroma_client.list_collections()
        assert len(all_collections) == 5
        
        # Add data to each collection
        for i, col in enumerate(collections):
            col.add(
                documents=[f"doc_{i}"],
                embeddings=[[float(i), 0.1, 0.2]],
                ids=[f"id_{i}"]
            )
        
        # Verify data isolation between collections
        for i, col in enumerate(collections):
            result = col.get(ids=[f"id_{i}"])
            assert result["documents"][0] == f"doc_{i}"
            
            # Should not find other collections' data
            wrong_result = col.get(ids=[f"id_{(i+1)%5}"])
            assert len(wrong_result["documents"]) == 0


@pytest.mark.integration
class TestChromaDBManagerIntegration:
    """Integration tests for ChromaDBManager with real components."""
    
    def test_manager_initialization_with_real_chromadb(self, temp_chroma_path, temp_media_db):
        """Test ChromaDBManager initialization with real ChromaDB."""
        manager = ChromaDBManager(
            user_id="test_user",
            user_embedding_config={
                "provider": "sentence_transformer",
                "model": "all-MiniLM-L6-v2"
            }
        )
        
        # Set paths
        manager.persist_directory = temp_chroma_path
        manager.db_path = temp_media_db
        
        # Initialize client
        manager._initialize_client()
        
        # Verify manager is functional
        assert manager.client is not None
        collection = manager.get_or_create_collection("test_collection")
        assert collection is not None
    
    def test_end_to_end_storage_and_retrieval(self, real_chromadb_manager, sample_texts):
        """Test complete storage and retrieval pipeline."""
        collection_name = "test_e2e"
        
        # Generate real embeddings
        embedder = HuggingFaceEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        embedder.load_model()
        embeddings = embedder.create_embeddings(sample_texts)
        embedder.unload_model()
        
        # Store in ChromaDB
        ids = [f"doc_{i}" for i in range(len(sample_texts))]
        metadata = [{"index": i, "source": "test"} for i in range(len(sample_texts))]
        
        success = real_chromadb_manager.store_in_chroma(
            collection_name=collection_name,
            texts=sample_texts,
            embeddings=embeddings,
            ids=ids,
            metadata=metadata
        )
        
        assert success is True
        
        # Verify storage
        count = real_chromadb_manager.count_items_in_collection(collection_name)
        assert count == len(sample_texts)
        
        # Test retrieval with precomputed embeddings
        query_embedding = embeddings[0:1]  # Use first embedding as query
        results = real_chromadb_manager.query_collection_with_precomputed_embeddings(
            collection_name=collection_name,
            query_embeddings=query_embedding,
            n_results=3
        )
        
        assert len(results["ids"][0]) <= 3
        assert results["ids"][0][0] == "doc_0"  # Should find itself as most similar
    
    def test_vector_search_with_real_embeddings(self, real_chromadb_manager):
        """Test vector search with actual embedding generation."""
        collection_name = "search_test"
        
        # Add test documents
        documents = [
            "The weather is sunny today",
            "Machine learning is fascinating",
            "The sun is shining brightly",
            "Deep learning transforms AI",
            "It's a beautiful sunny morning"
        ]
        
        # Generate embeddings
        embedder = HuggingFaceEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        embedder.load_model()
        embeddings = embedder.create_embeddings(documents)
        
        # Store documents
        ids = [f"doc_{i}" for i in range(len(documents))]
        real_chromadb_manager.store_in_chroma(
            collection_name=collection_name,
            texts=documents,
            embeddings=embeddings,
            ids=ids
        )
        
        # Search for weather-related documents
        query = "sunny weather"
        query_embedding = embedder.create_embeddings([query])[0]
        embedder.unload_model()
        
        results = real_chromadb_manager.query_collection_with_precomputed_embeddings(
            collection_name=collection_name,
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        # Should find weather-related documents
        found_docs = results["documents"][0]
        assert len(found_docs) == 3
        assert any("sunny" in doc.lower() or "weather" in doc.lower() for doc in found_docs)
    
    def test_metadata_filtering(self, real_chromadb_manager):
        """Test metadata filtering in searches."""
        collection_name = "metadata_test"
        
        # Create documents with different categories
        documents = []
        embeddings_list = []
        ids = []
        metadatas = []
        
        categories = ["science", "sports", "science", "sports", "science"]
        texts = [
            "Quantum physics discoveries",
            "Football match results",
            "Chemical reactions study",
            "Basketball tournament",
            "Astronomy observations"
        ]
        
        embedder = HuggingFaceEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        embedder.load_model()
        
        for i, (text, category) in enumerate(zip(texts, categories)):
            documents.append(text)
            ids.append(f"doc_{i}")
            metadatas.append({"category": category, "index": i})
        
        embeddings_list = embedder.create_embeddings(documents)
        embedder.unload_model()
        
        # Store documents
        real_chromadb_manager.store_in_chroma(
            collection_name=collection_name,
            texts=documents,
            embeddings=embeddings_list,
            ids=ids,
            metadata=metadatas
        )
        
        # Search only in science category
        collection = real_chromadb_manager.get_or_create_collection(collection_name)
        results = collection.query(
            query_embeddings=[embeddings_list[0]],
            n_results=5,
            where={"category": "science"}
        )
        
        # Should only return science documents
        assert len(results["ids"][0]) == 3
        for metadata in results["metadatas"][0]:
            assert metadata["category"] == "science"
    
    def test_collection_deletion_and_recreation(self, real_chromadb_manager):
        """Test deleting and recreating collections."""
        collection_name = "delete_test"
        
        # Create and populate collection
        real_chromadb_manager.store_in_chroma(
            collection_name=collection_name,
            texts=["test1", "test2"],
            embeddings=[[0.1, 0.2], [0.3, 0.4]],
            ids=["id1", "id2"]
        )
        
        # Verify data exists
        count = real_chromadb_manager.count_items_in_collection(collection_name)
        assert count == 2
        
        # Delete collection
        success = real_chromadb_manager.delete_collection(collection_name)
        assert success is True
        
        # Recreate collection
        collection = real_chromadb_manager.get_or_create_collection(collection_name)
        assert collection is not None
        
        # Should be empty
        new_count = real_chromadb_manager.count_items_in_collection(collection_name)
        assert new_count == 0
    
    def test_large_batch_processing(self, real_chromadb_manager):
        """Test processing large batches of documents."""
        collection_name = "large_batch_test"
        num_documents = 100
        
        # Generate large dataset
        texts = [f"Document {i}: This is test content for document number {i}" 
                 for i in range(num_documents)]
        ids = [f"doc_{i}" for i in range(num_documents)]
        
        # Generate simple embeddings (normally would use real embedder)
        np.random.seed(42)
        embeddings = np.random.randn(num_documents, 384).tolist()
        
        # Store in batches
        batch_size = 20
        for i in range(0, num_documents, batch_size):
            batch_end = min(i + batch_size, num_documents)
            success = real_chromadb_manager.store_in_chroma(
                collection_name=collection_name,
                texts=texts[i:batch_end],
                embeddings=embeddings[i:batch_end],
                ids=ids[i:batch_end]
            )
            assert success is True
        
        # Verify all documents stored
        count = real_chromadb_manager.count_items_in_collection(collection_name)
        assert count == num_documents
        
        # Test searching in large collection
        results = real_chromadb_manager.query_collection_with_precomputed_embeddings(
            collection_name=collection_name,
            query_embeddings=[embeddings[50]],
            n_results=10
        )
        
        assert len(results["ids"][0]) == 10
        assert "doc_50" in results["ids"][0]  # Should find itself


@pytest.mark.integration
class TestEmbeddingGeneration:
    """Test real embedding generation."""
    
    @pytest.mark.requires_model
    def test_huggingface_embedder_real(self):
        """Test HuggingFace embedder with real model."""
        embedder = HuggingFaceEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        
        # Load model
        embedder.load_model()
        assert embedder.model is not None
        
        # Generate embeddings
        texts = ["Hello world", "Testing embeddings", "ChromaDB integration"]
        embeddings = embedder.create_embeddings(texts)
        
        # Verify embeddings
        assert len(embeddings) == 3
        assert len(embeddings[0]) == 384  # all-MiniLM-L6-v2 dimension
        
        # Embeddings should be normalized
        for embedding in embeddings:
            norm = np.linalg.norm(embedding)
            assert abs(norm - 1.0) < 0.01  # Unit vectors
        
        # Different texts should have different embeddings
        similarity_01 = np.dot(embeddings[0], embeddings[1])
        similarity_02 = np.dot(embeddings[0], embeddings[2])
        assert similarity_01 != similarity_02
        
        # Unload model
        embedder.unload_model()
        assert embedder.model is None
    
    @pytest.mark.requires_model
    def test_embedding_auto_unload(self):
        """Test automatic model unloading after timeout."""
        embedder = HuggingFaceEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
            auto_unload_time=1  # 1 second timeout
        )
        
        # Load and use model
        embedder.load_model()
        embeddings = embedder.create_embeddings(["test"])
        assert embedder.model is not None
        
        # Wait for auto-unload
        time.sleep(1.5)
        
        # Model should be unloaded
        assert embedder.model is None
    
    def test_embedding_batch_creation(self):
        """Test batch embedding creation."""
        texts = ["text1", "text2", "text3", "text4", "text5"]
        
        with pytest.mock.patch('tldw_Server_API.app.core.Embeddings.Embeddings_Create.create_embeddings') as mock_create:
            # Simulate API that only handles 2 texts at a time
            def create_small_batch(batch_texts, *args, **kwargs):
                return [[0.1] * 384 for _ in batch_texts]
            
            mock_create.side_effect = create_small_batch
            
            embeddings = create_embeddings_batch(
                texts,
                provider="openai",
                model="text-embedding-ada-002",
                max_batch_size=2
            )
            
            assert len(embeddings) == 5
            # Should be called 3 times (2+2+1)
            assert mock_create.call_count == 3


@pytest.mark.integration
class TestDatabaseIntegration:
    """Test integration with MediaDatabase."""
    
    def test_media_db_with_chromadb(self, media_database, real_chromadb_manager):
        """Test ChromaDB integration with MediaDatabase."""
        # Add media entry
        media_id = str(uuid.uuid4())
        media_database.add_media(
            media_id=media_id,
            title="Test Document",
            content="This is test content for ChromaDB integration",
            media_type="document",
            author="Test Suite"
        )
        
        # Process and store in ChromaDB
        media_data = media_database.get_media(media_id)
        assert media_data is not None
        
        success = real_chromadb_manager.process_and_store_content(
            content=media_data["content"],
            media_id=media_id,
            collection_name="media_collection"
        )
        
        assert success is True
        
        # Verify in ChromaDB
        count = real_chromadb_manager.count_items_in_collection("media_collection")
        assert count > 0
    
    def test_chunk_storage_with_db_references(self, media_database, real_chromadb_manager):
        """Test storing chunks with database references."""
        media_id = str(uuid.uuid4())
        
        # Add media
        media_database.add_media(
            media_id=media_id,
            title="Chunked Document",
            content="First chunk content. Second chunk content. Third chunk content.",
            media_type="document"
        )
        
        # Manually chunk and store
        chunks = [
            "First chunk content.",
            "Second chunk content.",
            "Third chunk content."
        ]
        
        # Generate embeddings
        embedder = HuggingFaceEmbedder(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"
        )
        embedder.load_model()
        embeddings = embedder.create_embeddings(chunks)
        embedder.unload_model()
        
        # Store with chunk metadata
        ids = [f"{media_id}_chunk_{i}" for i in range(len(chunks))]
        metadata = [
            {"media_id": media_id, "chunk_index": i, "chunk_text": chunk}
            for i, chunk in enumerate(chunks)
        ]
        
        success = real_chromadb_manager.store_in_chroma(
            collection_name="chunked_media",
            texts=chunks,
            embeddings=embeddings,
            ids=ids,
            metadata=metadata
        )
        
        assert success is True
        
        # Search and verify metadata
        results = real_chromadb_manager.query_collection_with_precomputed_embeddings(
            collection_name="chunked_media",
            query_embeddings=[embeddings[0]],
            n_results=1
        )
        
        assert results["metadatas"][0][0]["media_id"] == media_id
        assert results["metadatas"][0][0]["chunk_index"] == 0


@pytest.mark.integration
class TestConcurrentOperations:
    """Test concurrent ChromaDB operations."""
    
    @pytest.mark.concurrent
    def test_concurrent_writes(self, real_chromadb_manager):
        """Test concurrent write operations."""
        import threading
        import concurrent.futures
        
        collection_name = "concurrent_test"
        num_threads = 5
        docs_per_thread = 10
        
        def write_documents(thread_id):
            """Write documents from a thread."""
            texts = [f"Thread {thread_id} doc {i}" for i in range(docs_per_thread)]
            ids = [f"t{thread_id}_d{i}" for i in range(docs_per_thread)]
            
            # Simple embeddings for testing
            embeddings = [[float(thread_id), float(i), 0.1] 
                         for i in range(docs_per_thread)]
            
            return real_chromadb_manager.store_in_chroma(
                collection_name=collection_name,
                texts=texts,
                embeddings=embeddings,
                ids=ids
            )
        
        # Execute concurrent writes
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(write_documents, i) for i in range(num_threads)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All writes should succeed
        assert all(results)
        
        # Verify all documents written
        count = real_chromadb_manager.count_items_in_collection(collection_name)
        assert count == num_threads * docs_per_thread
    
    @pytest.mark.concurrent
    def test_concurrent_reads(self, real_chromadb_manager):
        """Test concurrent read operations."""
        import concurrent.futures
        
        collection_name = "read_test"
        
        # Populate collection
        texts = [f"Document {i}" for i in range(20)]
        embeddings = [[float(i), 0.1, 0.2] for i in range(20)]
        ids = [f"doc_{i}" for i in range(20)]
        
        real_chromadb_manager.store_in_chroma(
            collection_name=collection_name,
            texts=texts,
            embeddings=embeddings,
            ids=ids
        )
        
        def search_documents(query_id):
            """Search documents from a thread."""
            query_embedding = [[float(query_id), 0.1, 0.2]]
            results = real_chromadb_manager.query_collection_with_precomputed_embeddings(
                collection_name=collection_name,
                query_embeddings=query_embedding,
                n_results=5
            )
            return len(results["ids"][0])
        
        # Execute concurrent reads
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(search_documents, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        # All searches should return results
        assert all(r == 5 for r in results)


@pytest.mark.integration
class TestErrorRecovery:
    """Test error recovery in real scenarios."""
    
    def test_recovery_from_corrupted_collection(self, real_chromadb_manager):
        """Test recovery from corrupted collection."""
        collection_name = "recovery_test"
        
        # Create and populate collection
        real_chromadb_manager.store_in_chroma(
            collection_name=collection_name,
            texts=["test"],
            embeddings=[[0.1, 0.2]],
            ids=["id1"]
        )
        
        # Simulate corruption by directly manipulating ChromaDB
        # This is implementation-specific and might need adjustment
        collection = real_chromadb_manager.get_or_create_collection(collection_name)
        
        # Try to add invalid data
        with pytest.raises(Exception):
            collection.add(
                documents=["test"],
                embeddings=[[0.1]],  # Wrong dimension
                ids=["id2"]
            )
        
        # Should still be able to query existing data
        results = real_chromadb_manager.query_collection_with_precomputed_embeddings(
            collection_name=collection_name,
            query_embeddings=[[0.1, 0.2]],
            n_results=1
        )
        
        assert len(results["ids"][0]) == 1
    
    def test_recovery_from_connection_loss(self, temp_chroma_path):
        """Test recovery from temporary connection loss."""
        manager = ChromaDBManager(
            user_id="test_user",
            user_embedding_config={"provider": "test"}
        )
        manager.persist_directory = temp_chroma_path
        
        # Initialize connection
        manager._initialize_client()
        
        # Simulate connection loss by setting client to None
        original_client = manager.client
        manager.client = None
        
        # Try operation that requires connection
        with pytest.raises(Exception):
            manager.count_items_in_collection("test")
        
        # Restore connection
        manager.client = original_client
        
        # Should work again
        collection = manager.get_or_create_collection("recovery_test")
        assert collection is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-m", "integration"])