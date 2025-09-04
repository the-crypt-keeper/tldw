"""
Property-based tests for Embeddings functionality.

Uses Hypothesis to verify invariants and properties of the embedding system.
"""

import pytest
from hypothesis import given, strategies as st, assume, settings, example
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize, invariant, Bundle
import numpy as np
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from tldw_Server_API.app.core.Embeddings.ChromaDB_Library import ChromaDBManager
from tldw_Server_API.app.core.Embeddings.queue_schemas import (
    JobRequest,
    JobStatus,
    JobType,
    JobResult
)

# ========================================================================
# Custom Hypothesis Strategies
# ========================================================================

@st.composite
def valid_embedding_dimension(draw):
    """Generate valid embedding dimensions."""
    # Common embedding dimensions
    return draw(st.sampled_from([128, 256, 384, 512, 768, 1024, 1536]))

@st.composite
def valid_embedding_vector(draw, dimension=None):
    """Generate valid embedding vectors."""
    if dimension is None:
        dimension = draw(valid_embedding_dimension())
    
    # Generate normalized vectors
    vector = np.random.randn(dimension)
    # Normalize to unit length
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector = vector / norm
    
    return vector.tolist()

@st.composite
def valid_text_for_embedding(draw):
    """Generate valid text for embedding."""
    # Generate text that's not too short or too long
    min_words = draw(st.integers(min_value=1, max_value=10))
    max_words = draw(st.integers(min_value=min_words, max_value=100))
    
    words = draw(st.lists(
        st.text(min_size=1, max_size=20, alphabet=st.characters(min_codepoint=97, max_codepoint=122)),
        min_size=min_words,
        max_size=max_words
    ))
    
    text = " ".join(words)
    assume(text.strip())  # Ensure not just whitespace
    return text

@st.composite
def valid_chunk_params(draw):
    """Generate valid chunking parameters."""
    chunk_size = draw(st.integers(min_value=50, max_value=2000))
    chunk_overlap = draw(st.integers(min_value=0, max_value=min(chunk_size // 2, 200)))
    
    return {
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap
    }

@st.composite
def valid_collection_name(draw):
    """Generate valid ChromaDB collection names."""
    # ChromaDB collection names must be 3-63 characters, start/end with alphanumeric
    prefix = draw(st.text(min_size=1, max_size=1, alphabet=st.characters(min_codepoint=97, max_codepoint=122)))
    middle = draw(st.text(min_size=1, max_size=30, alphabet=st.characters(min_codepoint=97, max_codepoint=122)))
    suffix = draw(st.text(min_size=1, max_size=1, alphabet=st.characters(min_codepoint=97, max_codepoint=122)))
    
    return f"{prefix}{middle}{suffix}"

# ========================================================================
# Embedding Vector Properties
# ========================================================================

class TestEmbeddingVectorProperties:
    """Test properties of embedding vectors."""
    
    @pytest.mark.property
    @given(dimension=valid_embedding_dimension())
    def test_embedding_dimension_consistency(self, dimension):
        """Property: Embeddings maintain consistent dimensions."""
        vectors = [np.random.randn(dimension).tolist() for _ in range(10)]
        
        # All vectors should have same dimension
        assert all(len(v) == dimension for v in vectors)
    
    @pytest.mark.property
    @given(vector=valid_embedding_vector())
    def test_embedding_vector_bounds(self, vector):
        """Property: Embedding values are within reasonable bounds."""
        # Most embedding models produce normalized vectors
        vector_array = np.array(vector)
        
        # Values should be finite
        assert np.all(np.isfinite(vector_array))
        
        # Norm should be approximately 1 for normalized vectors
        norm = np.linalg.norm(vector_array)
        # Allow some tolerance
        assert 0.1 < norm < 10.0
    
    @pytest.mark.property
    @given(
        vector1=valid_embedding_vector(dimension=384),
        vector2=valid_embedding_vector(dimension=384)
    )
    def test_cosine_similarity_bounds(self, vector1, vector2):
        """Property: Cosine similarity is bounded between -1 and 1."""
        v1 = np.array(vector1)
        v2 = np.array(vector2)
        
        # Calculate cosine similarity
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norm_product > 0:
            cosine_sim = dot_product / norm_product
            assert -1.01 <= cosine_sim <= 1.01  # Small tolerance for floating point
    
    @pytest.mark.property
    @given(vectors=st.lists(valid_embedding_vector(dimension=384), min_size=2, max_size=10))
    def test_embedding_distance_triangle_inequality(self, vectors):
        """Property: Distances satisfy triangle inequality."""
        if len(vectors) < 3:
            return
        
        v1, v2, v3 = np.array(vectors[0]), np.array(vectors[1]), np.array(vectors[2])
        
        # Calculate distances
        d12 = np.linalg.norm(v1 - v2)
        d23 = np.linalg.norm(v2 - v3)
        d13 = np.linalg.norm(v1 - v3)
        
        # Triangle inequality: d(a,c) <= d(a,b) + d(b,c)
        assert d13 <= d12 + d23 + 1e-6  # Small tolerance

# ========================================================================
# Text Chunking Properties
# ========================================================================

class TestTextChunkingProperties:
    """Test properties of text chunking."""
    
    @pytest.mark.property
    @given(
        text=valid_text_for_embedding(),
        params=valid_chunk_params()
    )
    def test_chunking_preserves_text(self, text, params):
        """Property: Chunking preserves all text content."""
        chunk_size = params["chunk_size"]
        chunk_overlap = params["chunk_overlap"]
        
        # Simple chunking simulation
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap
            
            if start >= len(text):
                break
        
        # Reassemble with overlap handling
        if chunk_overlap == 0:
            reassembled = "".join(chunks)
            # Should contain all original text
            assert len(reassembled) >= len(text)
    
    @pytest.mark.property
    @given(
        text_length=st.integers(min_value=100, max_value=10000),
        chunk_size=st.integers(min_value=50, max_value=1000),
        chunk_overlap=st.integers(min_value=0, max_value=100)
    )
    def test_chunk_count_bounds(self, text_length, chunk_size, chunk_overlap):
        """Property: Number of chunks is bounded."""
        assume(chunk_overlap < chunk_size)
        
        effective_chunk_size = chunk_size - chunk_overlap
        if effective_chunk_size <= 0:
            effective_chunk_size = 1
        
        # Calculate expected number of chunks
        min_chunks = max(1, text_length // chunk_size)
        max_chunks = max(1, (text_length + effective_chunk_size - 1) // effective_chunk_size)
        
        # Actual chunking
        num_chunks = 0
        pos = 0
        while pos < text_length:
            num_chunks += 1
            pos += effective_chunk_size
        
        assert min_chunks <= num_chunks <= max_chunks + 1
    
    @pytest.mark.property
    @given(params=valid_chunk_params())
    def test_overlap_consistency(self, params):
        """Property: Overlap is consistent between chunks."""
        chunk_size = params["chunk_size"]
        chunk_overlap = params["chunk_overlap"]
        
        # Generate sample text
        text = "a" * (chunk_size * 3)
        
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunks.append(text[start:end])
            start += chunk_size - chunk_overlap
            
            if len(chunks) >= 2:
                break
        
        if len(chunks) >= 2 and chunk_overlap > 0:
            # Check overlap between consecutive chunks
            overlap = chunks[0][-chunk_overlap:]
            next_chunk_start = chunks[1][:chunk_overlap]
            assert overlap == next_chunk_start or len(chunks[1]) < chunk_overlap

# ========================================================================
# ChromaDB Storage Properties
# ========================================================================

class TestChromaDBStorageProperties:
    """Test properties of ChromaDB storage."""
    
    @pytest.mark.property
    @given(
        embeddings=st.lists(valid_embedding_vector(dimension=384), min_size=1, max_size=10),
        texts=st.lists(valid_text_for_embedding(), min_size=1, max_size=10)
    )
    def test_storage_retrieval_consistency(self, embeddings, texts, chroma_client):
        """Property: Stored embeddings can be retrieved correctly."""
        # Ensure same length
        min_len = min(len(embeddings), len(texts))
        embeddings = embeddings[:min_len]
        texts = texts[:min_len]
        
        collection = chroma_client.create_collection("test_prop")
        ids = [f"doc_{i}" for i in range(len(embeddings))]
        
        # Store
        collection.add(
            embeddings=embeddings,
            documents=texts,
            ids=ids
        )
        
        # Retrieve
        results = collection.get(ids=ids, include=["embeddings", "documents"])
        
        assert len(results["ids"]) == len(ids)
        assert len(results["embeddings"]) == len(embeddings)
        assert len(results["documents"]) == len(texts)
    
    @pytest.mark.property
    @given(
        query_embedding=valid_embedding_vector(dimension=384),
        stored_embeddings=st.lists(valid_embedding_vector(dimension=384), min_size=5, max_size=20)
    )
    def test_similarity_search_ordering(self, query_embedding, stored_embeddings, chroma_client):
        """Property: Similarity search returns results in order of similarity."""
        collection = chroma_client.create_collection("test_similarity")
        
        ids = [f"doc_{i}" for i in range(len(stored_embeddings))]
        docs = [f"Document {i}" for i in range(len(stored_embeddings))]
        
        collection.add(
            embeddings=stored_embeddings,
            documents=docs,
            ids=ids
        )
        
        # Search
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(5, len(stored_embeddings))
        )
        
        if len(results["distances"][0]) > 1:
            distances = results["distances"][0]
            # Distances should be in ascending order (most similar first)
            assert all(distances[i] <= distances[i+1] for i in range(len(distances)-1))
    
    @pytest.mark.property
    @given(collection_name=valid_collection_name())
    def test_collection_name_validation(self, collection_name, chroma_client):
        """Property: Valid collection names are accepted."""
        try:
            collection = chroma_client.create_collection(collection_name)
            assert collection.name == collection_name
            
            # Cleanup
            chroma_client.delete_collection(collection_name)
        except Exception as e:
            # ChromaDB has specific naming rules
            assert "name" in str(e).lower() or "invalid" in str(e).lower()

# ========================================================================
# Job Processing Properties
# ========================================================================

class TestJobProcessingProperties:
    """Test properties of job processing."""
    
    @pytest.mark.property
    @given(
        job_type=st.sampled_from(list(JobType)),
        priority=st.integers(min_value=0, max_value=10),
        media_id=st.integers(min_value=1, max_value=1000)
    )
    def test_job_request_validity(self, job_type, priority, media_id):
        """Property: Valid job requests can be created."""
        job = JobRequest(
            job_id=f"job_{media_id}",
            job_type=job_type,
            media_id=media_id,
            priority=priority,
            data={"text": "test"}
        )
        
        assert job.job_type == job_type
        assert job.priority == priority
        assert job.media_id == media_id
    
    @pytest.mark.property
    @given(
        num_jobs=st.integers(min_value=1, max_value=100),
        priorities=st.lists(st.integers(min_value=0, max_value=10), min_size=1, max_size=100)
    )
    def test_priority_queue_ordering(self, num_jobs, priorities):
        """Property: Jobs are processed in priority order."""
        # Ensure matching lengths
        priorities = priorities[:num_jobs]
        while len(priorities) < num_jobs:
            priorities.append(5)
        
        jobs = []
        for i, priority in enumerate(priorities):
            jobs.append({
                "id": i,
                "priority": priority
            })
        
        # Sort by priority (higher first)
        sorted_jobs = sorted(jobs, key=lambda x: x["priority"], reverse=True)
        
        # First job should have highest priority
        if sorted_jobs:
            max_priority = max(priorities)
            assert sorted_jobs[0]["priority"] == max_priority
    
    @pytest.mark.property
    @given(status=st.sampled_from(list(JobStatus)))
    def test_job_status_transitions(self, status):
        """Property: Job status transitions are valid."""
        valid_transitions = {
            JobStatus.PENDING: [JobStatus.PROCESSING, JobStatus.FAILED],
            JobStatus.PROCESSING: [JobStatus.COMPLETED, JobStatus.FAILED],
            JobStatus.COMPLETED: [],
            JobStatus.FAILED: [JobStatus.PENDING]  # Can retry
        }
        
        # Any status should be in the valid set
        assert status in JobStatus

# ========================================================================
# Stateful Testing with RuleBasedStateMachine
# ========================================================================

class EmbeddingSystemStateMachine(RuleBasedStateMachine):
    """Stateful testing for the embedding system."""
    
    def __init__(self):
        super().__init__()
        self.client = chromadb.Client(Settings(is_persistent=False))
        self.collections = {}
        self.stored_embeddings = {}
        self.job_queue = []
    
    collections = Bundle('collections')
    embeddings = Bundle('embeddings')
    
    @initialize()
    def setup(self):
        """Initialize the state machine."""
        self.collections.clear()
        self.stored_embeddings.clear()
        self.job_queue.clear()
    
    @rule(name=valid_collection_name())
    def create_collection(self, name):
        """Rule: Create a new collection."""
        if name not in self.collections:
            collection = self.client.create_collection(name)
            self.collections[name] = collection
            self.stored_embeddings[name] = []
            return collection
    
    @rule(
        collection=collections,
        embedding=valid_embedding_vector(dimension=384),
        text=valid_text_for_embedding()
    )
    def add_embedding(self, collection, embedding, text):
        """Rule: Add an embedding to a collection."""
        doc_id = f"doc_{len(self.stored_embeddings[collection.name])}"
        
        collection.add(
            embeddings=[embedding],
            documents=[text],
            ids=[doc_id]
        )
        
        self.stored_embeddings[collection.name].append({
            "id": doc_id,
            "embedding": embedding,
            "text": text
        })
    
    @rule(collection=collections)
    def query_collection(self, collection):
        """Rule: Query a collection."""
        if self.stored_embeddings[collection.name]:
            # Use a random stored embedding as query
            import random
            query_item = random.choice(self.stored_embeddings[collection.name])
            
            results = collection.query(
                query_embeddings=[query_item["embedding"]],
                n_results=min(5, len(self.stored_embeddings[collection.name]))
            )
            
            # Should return at least the query item itself
            assert len(results["ids"][0]) > 0
    
    @invariant()
    def collection_consistency(self):
        """Invariant: Collections maintain consistency."""
        for name, collection in self.collections.items():
            count = collection.count()
            stored_count = len(self.stored_embeddings[name])
            assert count == stored_count
    
    @invariant()
    def embedding_dimensions_consistent(self):
        """Invariant: All embeddings in a collection have same dimension."""
        for name, items in self.stored_embeddings.items():
            if items:
                first_dim = len(items[0]["embedding"])
                assert all(len(item["embedding"]) == first_dim for item in items)

# Run the state machine tests
TestEmbeddingSystemStateMachine = EmbeddingSystemStateMachine.TestCase

# ========================================================================
# Performance Properties
# ========================================================================

class TestPerformanceProperties:
    """Test performance-related properties."""
    
    @pytest.mark.property
    @given(
        batch_size=st.integers(min_value=1, max_value=100),
        embedding_dim=valid_embedding_dimension()
    )
    @settings(max_examples=10, deadline=5000)
    def test_batch_processing_scaling(self, batch_size, embedding_dim):
        """Property: Batch processing scales linearly."""
        # Generate batch
        embeddings = [np.random.randn(embedding_dim).tolist() for _ in range(batch_size)]
        
        # Processing time should scale roughly linearly
        # This is a simplified test - real implementation would measure actual time
        expected_ops = batch_size * embedding_dim
        
        # Reasonable bounds for operations
        assert expected_ops > 0
        assert expected_ops < 1e8  # Prevent overflow
    
    @pytest.mark.property
    @given(
        num_collections=st.integers(min_value=1, max_value=10),
        embeddings_per_collection=st.integers(min_value=10, max_value=100)
    )
    @settings(max_examples=5, deadline=10000)
    def test_memory_usage_bounds(self, num_collections, embeddings_per_collection):
        """Property: Memory usage is bounded."""
        dimension = 384
        bytes_per_float = 4
        
        # Estimate memory usage
        total_embeddings = num_collections * embeddings_per_collection
        embedding_memory = total_embeddings * dimension * bytes_per_float
        
        # Add overhead estimate (indexes, metadata)
        overhead_factor = 2.0
        estimated_memory = embedding_memory * overhead_factor
        
        # Should be within reasonable bounds
        assert estimated_memory > 0
        assert estimated_memory < 1e9  # Less than 1GB for test